"""
Google Earth Engine Service
Handles GEE initialization and satellite data operations
"""
import ee
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List

class GEEService:
    """Service class for Google Earth Engine operations"""
    
    def __init__(self, project_id: str, service_account_key: str = ''):
        self.project_id = project_id
        self.service_account_key = service_account_key
        self.initialized = False
        self.logger = logging.getLogger(__name__)
    
    def _resolve_key_path(self) -> Optional[str]:
        """Resolve GEE service account key — supports file path or inline JSON string.
        
        On local dev: GEE_SERVICE_ACCOUNT_KEY = path/to/key.json
        On Render:    GEE_SERVICE_ACCOUNT_KEY = {"type":"service_account",...}  (inline JSON)
        """
        if not self.service_account_key:
            return None
        
        # Already a valid file path
        if os.path.exists(self.service_account_key):
            return self.service_account_key
        
        # Might be inline JSON string — write to a temp file
        if self.service_account_key.strip().startswith('{'):
            import json, tempfile
            try:
                key_dict = json.loads(self.service_account_key)
                # HF Spaces (and some platforms) double-escape \n inside env var values.
                # Ensure the PEM private key has real newline characters, not literal \\n.
                if 'private_key' in key_dict:
                    key_dict['private_key'] = key_dict['private_key'].replace('\\n', '\n')
                tmp = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', prefix='gee_key_', delete=False
                )
                json.dump(key_dict, tmp)
                tmp.close()
                self.logger.info(f"GEE key written to temp file: {tmp.name}")
                return tmp.name
            except json.JSONDecodeError:
                self.logger.error("GEE_SERVICE_ACCOUNT_KEY looks like JSON but failed to parse")
                return None
        
        self.logger.warning(f"GEE_SERVICE_ACCOUNT_KEY is not a valid file path or JSON string")
        return None
        
    def initialize(self) -> bool:
        """Initialize Google Earth Engine using service account credentials."""
        try:
            key_path = self._resolve_key_path()
            if key_path and os.path.exists(key_path):
                # ── Service account auth (preferred) ──
                import json
                with open(key_path) as f:
                    key_data = json.load(f)
                service_account = key_data.get('client_email', '')
                credentials = ee.ServiceAccountCredentials(service_account, key_path)
                ee.Initialize(credentials, project=self.project_id)
                self.logger.info(f"GEE initialized with service account: {service_account}")
                
                # Quick smoke test — make sure the credentials actually work
                try:
                    _ = ee.Number(1).getInfo()
                    self.logger.info("GEE smoke test passed (ee.Number(1).getInfo() OK)")
                except Exception as smoke_err:
                    self.logger.error(f"GEE smoke test FAILED: {smoke_err}")
                    self.initialized = False
                    return False
            else:
                # ── Fallback: default / user credentials ──
                try:
                    ee.Initialize(project=self.project_id)
                    self.logger.info("GEE initialized with default credentials")
                except Exception:
                    ee.Authenticate()
                    ee.Initialize(project=self.project_id)
                    self.logger.info("GEE initialized after interactive auth")
            
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"GEE initialization failed: {str(e)}")
            return False
    
    def get_satellite_data(self, 
                          latitude: float, 
                          longitude: float, 
                          start_date: str, 
                          end_date: str,
                          collection: str = 'COPERNICUS/S2_SR',
                          cloud_filter: int = 20) -> Dict[str, Any]:
        """
        Get satellite data for a specific location and date range
        
        Args:
            latitude: Target latitude
            longitude: Target longitude
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            collection: Satellite collection to use
            cloud_filter: Maximum cloud coverage percentage
            
        Returns:
            Dictionary containing satellite data information
        """
        if not self.initialized:
            raise Exception("GEE not initialized")
            
        try:
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            
            # Handle different asset types (Image vs ImageCollection)
            try:
                # Try as ImageCollection first (most common case)
                collection = ee.ImageCollection(collection) \
                    .filterBounds(point) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
                
                # Check if collection exists and has data
                collection_size = collection.size().getInfo()
                
                if collection_size == 0:
                    return {
                        'status': 'no_data',
                        'message': 'No satellite data available for the specified parameters',
                        'count': 0
                    }
                
                # Get the most recent image
                latest_image = collection.sort('system:time_start', False).first()
                
            except Exception as e:
                if "Expected asset" in str(e) and "ImageCollection" in str(e):
                    # Handle single Image assets (like SRTM elevation)
                    self.logger.info(f"Asset {collection} is an Image, not ImageCollection. Using direct Image access.")
                    latest_image = ee.Image(collection)
                    collection_size = 1
                else:
                    raise e
            
            # Get image properties
            try:
                properties = latest_image.getInfo()['properties']
            except:
                # Fallback for images without detailed properties
                properties = {
                    'system:time_start': 'N/A',
                    'CLOUDY_PIXEL_PERCENTAGE': 0,
                    'PRODUCT_ID': collection
                }
            
            # Calculate indices for the latest image
            indices_image = self._calculate_indices(latest_image)
            
            return {
                'status': 'success',
                'count': collection_size,
                'latest_image': {
                    'date': properties.get('system:time_start', 'N/A'),
                    'cloud_coverage': properties.get('CLOUDY_PIXEL_PERCENTAGE', 0),
                    'product_id': properties.get('PRODUCT_ID', collection)
                },
                'indices_available': True,
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting satellite data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_indices(self, image: ee.Image) -> ee.Image:
        """Calculate vegetation and water indices"""
        try:
            # NDVI (Normalized Difference Vegetation Index)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            # NDWI (Normalized Difference Water Index)
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            
            # NBR (Normalized Burn Ratio)
            nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
            
            # Add indices to the original image
            return image.addBands([ndvi, ndwi, nbr])
            
        except Exception as e:
            self.logger.error(f"Error calculating indices: {str(e)}")
            return image
    
    def get_region_data(self, 
                       bounds: List[List[float]], 
                       start_date: str, 
                       end_date: str,
                       scale: int = 10) -> Dict[str, Any]:
        """
        Get satellite data for a region defined by bounds
        
        Args:
            bounds: List of [lon, lat] coordinates defining the region
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            scale: Pixel scale in meters
            
        Returns:
            Dictionary containing region data information
        """
        if not self.initialized:
            raise Exception("GEE not initialized")
            
        try:
            # Create polygon geometry
            polygon = ee.Geometry.Polygon(bounds)
            
            # Get satellite collection
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(polygon) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            collection_size = collection.size().getInfo()
            
            if collection_size == 0:
                return {
                    'status': 'no_data',
                    'message': 'No satellite data available for the specified region',
                    'count': 0
                }
            
            # Create median composite
            median_image = collection.median()
            
            # Calculate indices
            indices_image = self._calculate_indices(median_image)
            
            # Get region statistics
            stats = indices_image.select(['NDVI', 'NDWI', 'NBR']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            
            return {
                'status': 'success',
                'count': collection_size,
                'region_stats': stats,
                'composite_created': True,
                'bounds': bounds
            }
            
        except Exception as e:
            self.logger.error(f"Error getting region data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def check_data_availability(self, 
                               latitude: float, 
                               longitude: float, 
                               days_back: int = 30) -> Dict[str, Any]:
        """
        Check data availability for a location over the past N days
        
        Args:
            latitude: Target latitude
            longitude: Target longitude
            days_back: Number of days to check backwards
            
        Returns:
            Dictionary with availability information
        """
        if not self.initialized:
            raise Exception("GEE not initialized")
            
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            point = ee.Geometry.Point([longitude, latitude])
            
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(point) \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            total_images = collection.size().getInfo()
            
            # Filter for low cloud coverage
            low_cloud_collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            usable_images = low_cloud_collection.size().getInfo()
            
            return {
                'status': 'success',
                'total_images': total_images,
                'usable_images': usable_images,
                'usability_ratio': usable_images / total_images if total_images > 0 else 0,
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                },
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking data availability: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }