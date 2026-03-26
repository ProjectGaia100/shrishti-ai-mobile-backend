"""
GeoVision Fusion Prediction Service
Orchestrates data collection and runs the multi-model fusion pipeline
(LSTM MIMO + Tree Ensemble + Fusion Meta-Learner)
"""

import logging
import os
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from models.geovision_fusion_model import GeoVisionFusionModel
from services.weather_service import NASAPowerService
from services.feature_engineering_service import FeatureEngineeringService
from services.raster_data_service import RasterDataService

logger = logging.getLogger(__name__)


class GeoVisionFusionService:
    """Service for orchestrating GeoVision fusion predictions."""

    def __init__(self,
                 weather_service: Optional[NASAPowerService] = None,
                 feature_service: Optional[FeatureEngineeringService] = None,
                 raster_service: Optional[RasterDataService] = None,
                 satellite_service=None,
                 gee_service=None):
        """
        Initialize the GeoVision fusion service.

        Args:
            weather_service: Re-use existing NASA POWER service instance
            feature_service: Re-use existing feature engineering service instance
            raster_service:  Re-use existing raster data service instance
            satellite_service: Optional SatelliteImageService for CNN input
            gee_service: Optional GEEService (used to create satellite_service)
        """
        self.fusion_model = GeoVisionFusionModel()

        self.weather_service = weather_service or NASAPowerService()
        self.feature_service = feature_service or FeatureEngineeringService()
        self.raster_service = raster_service or RasterDataService()
        self.satellite_service = satellite_service
        self.gee_service = gee_service

        # Field name mappings (services return _perc, models expect _%)
        self.field_mappings = {
            'humidity_perc': 'humidity_%',
            'cloud_amount_perc': 'cloud_amount_%',
            'surface_soil_wetness_perc': 'surface_soil_wetness_%',
            'root_zone_soil_moisture_perc': 'root_zone_soil_moisture_%'
        }

        # Service statistics
        self.service_stats = {
            'service_start_time': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0.0,
            'models_loaded': False
        }

        logger.info("[GEOVISION_SVC] Fusion service initialized")

    # ────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ────────────────────────────────────────────────────────────────
    def initialize_service(self) -> Tuple[bool, str]:
        """Load all fusion pipeline models."""
        try:
            logger.info("[GEOVISION_SVC] Initializing fusion pipeline...")
            loaded = self.fusion_model.load_models()

            # Initialize satellite image service if GEE is available
            if self.satellite_service is None and self.gee_service is not None:
                try:
                    from services.satellite_image_service import SatelliteImageService
                    self.satellite_service = SatelliteImageService(
                        gee_service=self.gee_service,
                    )
                    sat_ok, sat_msg = self.satellite_service.initialize()
                    if sat_ok:
                        logger.info(f"[GEOVISION_SVC] Satellite image service initialized (GCS fallback)")
                    else:
                        logger.warning(f"[GEOVISION_SVC] Satellite service init warning: {sat_msg}")
                        self.satellite_service = None
                except Exception as e:
                    logger.warning(f"[GEOVISION_SVC] Satellite service setup failed (non-fatal): {e}")
                    self.satellite_service = None

            if loaded:
                self.service_stats['models_loaded'] = True
                return True, "GeoVision fusion pipeline initialized successfully"
            else:
                return False, "Failed to load minimum required models (need LSTM + Fusion)"
        except Exception as e:
            logger.error(f"[GEOVISION_SVC] Init error: {e}")
            return False, f"Initialization error: {str(e)}"

    # ────────────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────────────
    def _apply_field_mappings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map _perc → _% for model compatibility."""
        mapped = {}
        for key, value in data.items():
            mapped[self.field_mappings.get(key, key)] = value
        return mapped

    def validate_coordinates(self, latitude: float, longitude: float) -> Tuple[bool, str]:
        """Validate coordinate inputs."""
        try:
            lat, lon = float(latitude), float(longitude)
            if not (-90 <= lat <= 90):
                return False, f"Invalid latitude {lat}"
            if not (-180 <= lon <= 180):
                return False, f"Invalid longitude {lon}"
            return True, f"Coordinates validated: ({lat}, {lon})"
        except (ValueError, TypeError):
            return False, "Coordinates must be numeric"

    # ────────────────────────────────────────────────────────────────
    # DATA COLLECTION (same as HazardGuard)
    # ────────────────────────────────────────────────────────────────
    def _collect_weather_data(self, latitude: float, longitude: float,
                              reference_date: str) -> Tuple[bool, str, Optional[Dict]]:
        """Collect 60-day weather data from NASA POWER."""
        try:
            from models.weather_model import WeatherRequest
            weather_request = WeatherRequest(
                latitude=latitude,
                longitude=longitude,
                disaster_date=reference_date,
                days_before=60  # LSTM expects 60 timesteps
            )
            success, result = self.weather_service.fetch_weather_data(weather_request)
            if success:
                weather_data = result.get('weather_data', {})
                return True, "Weather data collected", weather_data
            else:
                return False, f"Weather fetch failed: {result.get('error', '?')}", None
        except Exception as e:
            return False, f"Weather error: {str(e)}", None

    def _collect_feature_data(self, weather_data: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """Engineer features from weather data."""
        try:
            success, result = self.feature_service.process_weather_features(
                weather_data=weather_data,
                event_duration=1.0,
                include_metadata=True
            )
            if success:
                return True, "Features engineered", result.get('engineered_features', {})
            else:
                return False, f"Feature engineering failed: {result.get('error', '?')}", None
        except Exception as e:
            return False, f"Feature error: {str(e)}", None

    def _collect_raster_data(self, latitude: float, longitude: float) -> Tuple[bool, str, Optional[Dict]]:
        """Collect raster/scalar features."""
        try:
            # Use the configured service path (env/local/remote) consistently.
            result = self.raster_service.process_raster_extraction([
                {
                    'longitude': float(longitude),
                    'latitude': float(latitude)
                }
            ])

            if result.get('success') and result.get('data'):
                feature_row = result['data'][0]
                # Strip coordinate keys; keep only scalar feature fields used by fusion.
                feature_row.pop('longitude', None)
                feature_row.pop('latitude', None)
                return True, "Raster data collected", feature_row

            return False, result.get('error', 'No raster data extracted'), {}
        except Exception as e:
            logger.warning(f"[GEOVISION_SVC] Raster error (non-fatal): {e}")
            return False, f"Raster error: {str(e)}", {}

    # ────────────────────────────────────────────────────────────────
    # MAIN PREDICTION
    # ────────────────────────────────────────────────────────────────
    def predict_for_location(self, latitude: float, longitude: float,
                             reference_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full GeoVision fusion prediction for a location.

        Args:
            latitude:  Location latitude
            longitude: Location longitude
            reference_date: Optional date (YYYY-MM-DD). Defaults to most recent
                            available NASA POWER date (~8 days ago).

        Returns:
            Comprehensive prediction result dict.
        """
        start_time = datetime.now()
        self.service_stats['total_requests'] += 1

        # Default to the most recent date NASA POWER reliably has data for.
        # NASA POWER has ~7-day lag; using 8 days to be safe.
        if not reference_date:
            safe_date = datetime.now() - timedelta(days=8)
            reference_date = safe_date.strftime('%Y-%m-%d')
            logger.info(f"[GEOVISION_SVC] Auto-computed reference_date: {reference_date} (most recent available)")

        # Validate coords
        is_valid, msg = self.validate_coordinates(latitude, longitude)
        if not is_valid:
            return {'success': False, 'error': msg}

        logger.info(f"[GEOVISION_SVC] Prediction for ({latitude}, {longitude}) date={reference_date}")

        data_collection = {
            'weather': {'success': False},
            'features': {'success': False},
            'raster': {'success': False}
        }

        # ── Collect weather ──
        wx_ok, wx_msg, weather_data = self._collect_weather_data(latitude, longitude, reference_date)
        data_collection['weather'] = {'success': wx_ok, 'message': wx_msg}
        if not wx_ok:
            self.service_stats['failed_predictions'] += 1
            return {
                'success': False,
                'error': f'Weather data collection failed: {wx_msg}',
                'data_collection': data_collection
            }

        # ── Engineer features ──
        feat_ok, feat_msg, feature_data = self._collect_feature_data(weather_data)
        data_collection['features'] = {'success': feat_ok, 'message': feat_msg}
        if not feat_ok:
            feature_data = {}  # Non-fatal — LSTM can still use raw weather

        # Apply field mappings (_perc → _%)
        weather_data = self._apply_field_mappings(weather_data)
        feature_data = self._apply_field_mappings(feature_data) if feature_data else {}

        # ── Collect raster ──
        raster_ok, raster_msg, raster_data = self._collect_raster_data(latitude, longitude)
        data_collection['raster'] = {'success': raster_ok, 'message': raster_msg}
        if not raster_ok:
            raster_data = {}  # Non-fatal — ensemble will use 0s

        # ── Fetch satellite imagery for CNN (optional, non-blocking) ──
        satellite_image = None
        if self.satellite_service is not None:
            try:
                logger.info("[GEOVISION_SVC] Fetching satellite imagery for CNN...")
                sat_result = self.satellite_service.fetch_and_preprocess(
                    latitude, longitude, reference_date
                )
                if sat_result.get('success') and sat_result.get('image') is not None:
                    satellite_image = sat_result['image']
                    data_collection['satellite'] = {
                        'success': True,
                        'message': 'Satellite imagery fetched',
                        'metadata': sat_result.get('metadata', {})
                    }
                    logger.info(f"[GEOVISION_SVC] Satellite image ready: {satellite_image.shape}")
                else:
                    data_collection['satellite'] = {
                        'success': False,
                        'message': sat_result.get('error', 'No imagery available')
                    }
                    logger.info(f"[GEOVISION_SVC] No satellite imagery: {sat_result.get('error')}")
            except Exception as e:
                data_collection['satellite'] = {'success': False, 'message': str(e)}
                logger.warning(f"[GEOVISION_SVC] Satellite fetch error (non-fatal): {e}")
        else:
            data_collection['satellite'] = {'success': False, 'message': 'Service not configured'}

        # ── Run fusion pipeline ──
        prediction = self.fusion_model.predict(weather_data, feature_data, raster_data, satellite_image)

        processing_time = (datetime.now() - start_time).total_seconds()

        if prediction.get('success'):
            self.service_stats['successful_predictions'] += 1
        else:
            self.service_stats['failed_predictions'] += 1

        return {
            'success': prediction.get('success', False),
            'error': prediction.get('error'),
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'reference_date': reference_date
            },
            'prediction': {
                'disaster_prediction': prediction.get('disaster_prediction'),
                'disaster_probabilities': prediction.get('disaster_probabilities'),
                'disaster_confidence': prediction.get('disaster_confidence'),
                'weather_prediction': prediction.get('weather_prediction'),
                'weather_probabilities': prediction.get('weather_probabilities'),
                'weather_confidence': prediction.get('weather_confidence'),
            },
            'intermediate': prediction.get('intermediate', {}),
            'metadata': prediction.get('metadata', {}),
            'data_collection': data_collection,
            'processing_time_seconds': round(processing_time, 2)
        }

    # ────────────────────────────────────────────────────────────────
    # STATUS / HEALTH
    # ────────────────────────────────────────────────────────────────
    def get_service_status(self) -> Dict[str, Any]:
        """Return service health status."""
        status = {
            'service': 'geovision_fusion',
            'models_loaded': self.service_stats['models_loaded'],
            'model_details': self.fusion_model.get_model_status(),
            'satellite_service': self.satellite_service is not None,
            'statistics': self.service_stats
        }
        if self.satellite_service is not None:
            try:
                status['satellite_details'] = self.satellite_service.get_status()
            except Exception:
                pass
        return status
