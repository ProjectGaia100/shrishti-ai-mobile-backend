"""
Satellite Image Service for GeoVision Fusion Pipeline
=====================================================
Fetches 6-band Sentinel-2 imagery (RGB + NDVI + NDWI + NBR) for CNN ResNet50 inference.

Pipeline:
  1. Query GEE for Sentinel-2 composite at the target location
  2. Export 6-band GeoTIFF to GCS bucket temp folder (via ee.batch.Export)
  3. Download from public GCS URL
  4. Preprocess: normalize, resize to (224, 224, 6)
  5. Return numpy array ready for CNN inference

Fast path (primary):
  - Uses ee.Image.getDownloadURL() for direct pixel download (no GCS round-trip)
  - Suitable for small tiles (224×224 at 10m = ~2.24km patch)

Slow path (fallback):
  - GEE export → GCS temp/ → download via public HTTPS URL
  - Uses service-account key or public bucket access
"""

import os
import io
import time
import json
import logging
import tempfile
import zipfile
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# 6 bands in the order the CNN expects
BAND_ORDER = ['B4', 'B3', 'B2', 'B8', 'B12']  # Raw bands needed for computation
CNN_INPUT_SIZE = 224
CNN_BANDS = 6  # R, G, B, NDVI, NDWI, NBR

# GCS bucket configuration
GCS_BUCKET = os.getenv('GCS_BUCKET', 'satellite-cog-data-for-shrishti')
GCS_TEMP_PREFIX = os.getenv('GCS_TEMP_PREFIX', 'temp')
GCS_PUBLIC_BASE = f'https://storage.googleapis.com/{GCS_BUCKET}'


class SatelliteImageService:
    """
    Fetches and preprocesses 6-band Sentinel-2 imagery for CNN inference.

    Uses GEE for image composition and either direct download or
    GCS export+download for the actual pixel data.
    """

    def __init__(self,
                 gee_service=None,
                 client_secret_path: Optional[str] = None,
                 download_dir: Optional[str] = None):
        """
        Args:
            gee_service: Existing GEEService instance (already initialized)
            client_secret_path: Path to Google OAuth2 client_secret.json (legacy, unused now)
            download_dir: Local directory for cached satellite images
        """
        self.gee_service = gee_service
        self.client_secret_path = client_secret_path or ""
        self.download_dir = download_dir or os.path.join(
            tempfile.gettempdir(), "geovision_satellite_cache"
        )
        os.makedirs(self.download_dir, exist_ok=True)

        self._gcs_client = None
        self._gcs_bucket = None

        self._initialized = False
        self._stats = {
            'images_fetched': 0,
            'direct_downloads': 0,
            'gcs_downloads': 0,
            'failures': 0,
        }

        logger.info("[SAT_IMG] Satellite image service created")

    # ──────────────────────────────────────────────────────
    # INITIALIZATION
    # ──────────────────────────────────────────────────────
    def initialize(self) -> Tuple[bool, str]:
        """Initialize the service (GEE must already be initialized)."""
        try:
            import ee
            # Verify GEE is initialized
            if self.gee_service and self.gee_service.initialized:
                logger.info("[SAT_IMG] Using existing GEE service")
            else:
                # Try to initialize GEE independently
                try:
                    ee.Initialize()
                    logger.info("[SAT_IMG] GEE initialized independently")
                except Exception:
                    return False, "GEE not initialized and no gee_service provided"

            # Initialize GCS client for fallback export path
            self._init_gcs_client()

            self._initialized = True
            return True, "Satellite image service initialized"
        except Exception as e:
            logger.error(f"[SAT_IMG] Initialization error: {e}")
            return False, str(e)

    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client for fallback export/download."""
        try:
            from google.cloud import storage as gcs_storage

            # Look for service account key (env var or auto-discover)
            sa_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
            if not sa_key:
                # Try well-known path next to the workspace root
                workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                for f in os.listdir(workspace_root):
                    if f.startswith('geovision-final') and f.endswith('.json'):
                        sa_key = os.path.join(workspace_root, f)
                        break

            if sa_key and os.path.exists(sa_key):
                self._gcs_client = gcs_storage.Client.from_service_account_json(sa_key)
                logger.info(f"[SAT_IMG] GCS client initialized from service account key")
            else:
                # Fall back to Application Default Credentials
                self._gcs_client = gcs_storage.Client()
                logger.info("[SAT_IMG] GCS client initialized with default credentials")

            self._gcs_bucket = self._gcs_client.bucket(GCS_BUCKET)
            logger.info(f"[SAT_IMG] GCS bucket '{GCS_BUCKET}' ready for export fallback")

        except Exception as e:
            logger.warning(f"[SAT_IMG] GCS client init failed (non-fatal): {e}")
            self._gcs_client = None
            self._gcs_bucket = None

    # ──────────────────────────────────────────────────────
    # MAIN PUBLIC METHOD
    # ──────────────────────────────────────────────────────
    def fetch_and_preprocess(self, latitude: float, longitude: float,
                             reference_date: Optional[str] = None,
                             lookback_days: int = 90,
                             cloud_max: int = 20) -> Dict[str, Any]:
        """
        Fetch a 6-band satellite image tile and preprocess for CNN inference.

        Args:
            latitude: Center latitude
            longitude: Center longitude
            reference_date: End date (YYYY-MM-DD). Defaults to ~8 days ago.
            lookback_days: How far back to search for cloud-free imagery
            cloud_max: Max cloud cover percentage

        Returns:
            {
                'success': bool,
                'image': np.ndarray (1, 224, 224, 6) or None,
                'metadata': { date, bands, source, ... },
                'error': str or None
            }
        """
        if not self._initialized:
            return {'success': False, 'image': None, 'error': 'Service not initialized'}

        try:
            import ee

            if not reference_date:
                reference_date = (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d')

            end_date = reference_date
            start_date = (datetime.strptime(reference_date, '%Y-%m-%d')
                          - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            logger.info(f"[SAT_IMG] Fetching imagery for ({latitude}, {longitude}), "
                        f"{start_date} to {end_date}")

            # Create AOI: ~2.24km box centered on point (224px × 10m = 2240m)
            half_side = 0.01  # ~1.1km in degrees at equator
            aoi = ee.Geometry.Rectangle([
                longitude - half_side, latitude - half_side,
                longitude + half_side, latitude + half_side
            ])

            # Build Sentinel-2 composite
            s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_max)))

            count = s2.size().getInfo()
            if count == 0:
                logger.warning(f"[SAT_IMG] No imagery found (cloud<{cloud_max}%)")
                return {'success': False, 'image': None,
                        'error': f'No Sentinel-2 imagery with <{cloud_max}% cloud in last {lookback_days} days'}

            logger.info(f"[SAT_IMG] Found {count} images, creating median composite")

            # Scale reflectance to 0-1
            def scale_s2(img):
                optical = img.select(['B2', 'B3', 'B4', 'B8', 'B12']).multiply(0.0001)
                return optical.copyProperties(img, img.propertyNames())

            composite = s2.map(scale_s2).median().clip(aoi)

            # Compute spectral indices
            ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI')
            nbr = composite.normalizedDifference(['B8', 'B12']).rename('NBR')

            # Stack: R(B4), G(B3), B(B2), NDVI, NDWI, NBR
            six_band = composite.select(['B4', 'B3', 'B2']).addBands([ndvi, ndwi, nbr])

            # ── Try direct download first (fast path) ──
            image_array = self._direct_download(six_band, aoi)

            if image_array is None and self._gcs_client:
                # ── Fallback: GEE → GCS temp/ → local ──
                logger.info("[SAT_IMG] Direct download failed, trying GCS export...")
                image_array = self._gcs_export_download(six_band, aoi, latitude, longitude)

            if image_array is None:
                return {'success': False, 'image': None,
                        'error': 'Failed to download satellite imagery (both direct and GCS paths)'}

            # Preprocess
            processed = self._preprocess(image_array)

            self._stats['images_fetched'] += 1
            return {
                'success': True,
                'image': processed,  # (1, 224, 224, 6)
                'metadata': {
                    'location': {'latitude': latitude, 'longitude': longitude},
                    'date_range': f'{start_date} to {end_date}',
                    'images_in_composite': count,
                    'bands': ['Red', 'Green', 'Blue', 'NDVI', 'NDWI', 'NBR'],
                    'shape': list(processed.shape),
                }
            }

        except Exception as e:
            self._stats['failures'] += 1
            logger.error(f"[SAT_IMG] Fetch error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'image': None, 'error': str(e)}

    # ──────────────────────────────────────────────────────
    # DIRECT DOWNLOAD (fast path, no GCS round-trip)
    # ──────────────────────────────────────────────────────
    def _direct_download(self, image, aoi) -> Optional[np.ndarray]:
        """Download pixels directly via ee.Image.getDownloadURL()."""
        try:
            import ee
            import requests

            url = image.getDownloadURL({
                'region': aoi,
                'scale': 10,
                'format': 'GEO_TIFF',
                'bands': ['B4', 'B3', 'B2', 'NDVI', 'NDWI', 'NBR'],
            })

            logger.info(f"[SAT_IMG] Direct download from GEE...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()

            # Read GeoTIFF from bytes
            import rasterio
            with rasterio.open(io.BytesIO(resp.content)) as src:
                img = src.read().astype(np.float32)  # (bands, H, W)
                img = np.transpose(img, (1, 2, 0))   # (H, W, bands)

            logger.info(f"[SAT_IMG] Direct download OK: shape={img.shape}")
            self._stats['direct_downloads'] += 1
            return img

        except Exception as e:
            logger.warning(f"[SAT_IMG] Direct download failed: {e}")
            return None

    # ──────────────────────────────────────────────────────
    # GCS EXPORT + DOWNLOAD (slow fallback path)
    # ──────────────────────────────────────────────────────
    def _gcs_export_download(self, image, aoi,
                              lat: float, lon: float) -> Optional[np.ndarray]:
        """Export to GCS temp/ folder, wait, download, delete."""
        try:
            import ee
            import requests

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"geovision_{lat:.4f}_{lon:.4f}_{timestamp}"
            gcs_path = f"{GCS_TEMP_PREFIX}/{filename}"

            # Start export task to Cloud Storage
            task = ee.batch.Export.image.toCloudStorage(
                image=image,
                description=filename,
                bucket=GCS_BUCKET,
                fileNamePrefix=gcs_path,
                region=aoi,
                scale=10,
                maxPixels=1e8,
                fileFormat='GeoTIFF',
            )
            task.start()
            logger.info(f"[SAT_IMG] GCS export started: gs://{GCS_BUCKET}/{gcs_path}")

            # Poll for completion (max ~5 minutes)
            max_wait = 300
            elapsed = 0
            while elapsed < max_wait:
                status = task.status()
                state = status.get('state', '')
                if state == 'COMPLETED':
                    logger.info(f"[SAT_IMG] Export completed in {elapsed}s")
                    break
                elif state in ('FAILED', 'CANCELLED'):
                    logger.error(f"[SAT_IMG] Export {state}: {status.get('error_message', '?')}")
                    return None
                time.sleep(10)
                elapsed += 10

            if elapsed >= max_wait:
                logger.warning("[SAT_IMG] Export timed out after 5 min")
                return None

            # Download from GCS public URL
            time.sleep(2)  # Brief pause for consistency
            tif_blob_name = f"{gcs_path}.tif"
            download_url = f"{GCS_PUBLIC_BASE}/{tif_blob_name}"

            logger.info(f"[SAT_IMG] Downloading from GCS: {download_url}")
            resp = requests.get(download_url, timeout=120)
            resp.raise_for_status()

            # Read GeoTIFF from bytes
            import rasterio
            with rasterio.open(io.BytesIO(resp.content)) as src:
                img = src.read().astype(np.float32)
                img = np.transpose(img, (1, 2, 0))

            # Cleanup: delete temp file from GCS
            try:
                if self._gcs_bucket:
                    blob = self._gcs_bucket.blob(tif_blob_name)
                    blob.delete()
                    logger.info(f"[SAT_IMG] Deleted temp file from GCS: {tif_blob_name}")
            except Exception as del_err:
                logger.warning(f"[SAT_IMG] Could not delete GCS temp file: {del_err}")

            self._stats['gcs_downloads'] += 1
            return img

        except Exception as e:
            logger.error(f"[SAT_IMG] GCS export/download failed: {e}")
            return None

    # ──────────────────────────────────────────────────────
    # PREPROCESSING
    # ──────────────────────────────────────────────────────
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess 6-band image for CNN ResNet50 inference.

        Expected input: (H, W, 6) with bands [Red, Green, Blue, NDVI, NDWI, NBR]
        Output: (1, 224, 224, 6) normalized to [0, 1]

        Normalization (matching training pipeline):
          - RGB (channels 0-2): clip to [0, 1]  (already 0-1 SR reflectance)
          - Indices (channels 3-5): clip [-1, 1] → rescale to [0, 1] via (x+1)/2
        """
        import tensorflow as tf

        # Handle NaN / Inf
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize RGB bands (0–2): clip to [0, 1]
        img[:, :, :3] = np.clip(img[:, :, :3], 0.0, 1.0)

        # Normalize index bands (3–5): clip [-1, 1] → rescale to [0, 1]
        img[:, :, 3:] = np.clip(img[:, :, 3:], -1.0, 1.0)
        img[:, :, 3:] = (img[:, :, 3:] + 1.0) / 2.0

        # Resize to (224, 224)
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_resized = tf.image.resize(img_tensor, (CNN_INPUT_SIZE, CNN_INPUT_SIZE),
                                       method='bilinear')
        img_np = img_resized.numpy()

        # Add batch dimension → (1, 224, 224, 6)
        return np.expand_dims(img_np, axis=0)

    # ──────────────────────────────────────────────────────
    # STATUS
    # ──────────────────────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Return service status."""
        return {
            'initialized': self._initialized,
            'gcs_client_available': self._gcs_client is not None,
            'gcs_bucket': GCS_BUCKET,
            'gcs_temp_prefix': GCS_TEMP_PREFIX,
            'download_dir': self.download_dir,
            'stats': self._stats,
        }
