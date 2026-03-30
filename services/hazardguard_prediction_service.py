"""
HazardGuard Disaster Prediction Service
Orchestrates data collection from weather, features, and raster services for disaster prediction
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from models.hazardguard_prediction_model import HazardGuardPredictionModel
from models.disaster_type_classifier_model import DisasterTypeClassifierModel
from services.weather_service import NASAPowerService
from services.feature_engineering_service import FeatureEngineeringService
from services.raster_data_service import RasterDataService

logger = logging.getLogger(__name__)

class HazardGuardPredictionService:
    """Service for orchestrating HazardGuard disaster predictions"""
    
    def __init__(self, weather_service: Optional[NASAPowerService] = None,
                 feature_service: Optional[FeatureEngineeringService] = None,
                 raster_service: Optional[RasterDataService] = None):
        """
        Initialize HazardGuard prediction service
        
        Args:
            weather_service: NASA POWER weather service instance
            feature_service: Feature engineering service instance  
            raster_service: Raster data service instance
        """
        # Initialize prediction models
        self.prediction_model = HazardGuardPredictionModel()
        self.disaster_type_classifier = DisasterTypeClassifierModel()  # NEW: Multi-stage classifier
        
        # Initialize or use provided services
        self.weather_service = weather_service or NASAPowerService()
        self.feature_service = feature_service or FeatureEngineeringService()
        self.raster_service = raster_service or RasterDataService()
        
        # Field name mappings from service format to model format
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
            'data_collection_failures': 0,
            'weather_fetch_failures': 0,
            'feature_engineering_failures': 0,
            'raster_fetch_failures': 0,
            'average_processing_time': 0.0,
            'model_loaded': False
        }
        
        logger.info("HazardGuard prediction service initialized")
    
    def initialize_service(self) -> Tuple[bool, str]:
        """
        Initialize the service by loading the prediction model
        
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("Initializing HazardGuard prediction service...")
            
            # Load combined binary prediction model
            model_loaded = self.prediction_model.load_model_components()
            
            if model_loaded:
                self.service_stats['model_loaded'] = True
                
                # Load disaster type classifiers (Storm, Flood, Drought, Landslide)
                logger.info("Loading disaster type classification models...")
                disaster_types_loaded = self.disaster_type_classifier.load_models()
                
                if disaster_types_loaded:
                    logger.info("[SUCCESS] All models loaded: Combined disaster + 4 type classifiers")
                else:
                    logger.warning("[PARTIAL] Combined model loaded but disaster type classifiers failed")
                
                logger.info("[SUCCESS] HazardGuard service initialization successful")
                return True, "Service initialized successfully"
            else:
                logger.error("[ERROR] Failed to load prediction model")
                return False, "Failed to load prediction model"
                
        except Exception as e:
            logger.error(f"Service initialization error: {e}")
            return False, f"Initialization error: {str(e)}"
    
    def _apply_field_mappings(self, data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Apply field name mappings for model compatibility
        Model expects _% suffix, but services return _perc suffix
        
        Args:
            data: Data dictionary with _perc suffix fields
            
        Returns:
            Data dictionary with _% suffix fields for model
        """
        mapped_data = {}
        for key, value in data.items():
            mapped_key = self.field_mappings.get(key, key)
            mapped_data[mapped_key] = value
        return mapped_data
    
    def validate_coordinates(self, latitude: float, longitude: float) -> Tuple[bool, str]:
        """
        Validate coordinate inputs
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            lat = float(latitude)
            lon = float(longitude)
            
            if not (-90 <= lat <= 90):
                return False, f"Invalid latitude {lat} (must be -90 to 90)"
            if not (-180 <= lon <= 180):
                return False, f"Invalid longitude {lon} (must be -180 to 180)"
            
            return True, f"Coordinates validated: ({lat}, {lon})"
            
        except (ValueError, TypeError):
            return False, "Coordinates must be numeric values"
    
    def collect_weather_data(self, latitude: float, longitude: float, 
                           start_date: str, end_date: str) -> Tuple[bool, str, Optional[Dict[str, List[float]]]]:
        """
        Collect weather data for the location and date range
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Tuple of (success, message, weather_data)
        """
        try:
            logger.debug(f"Collecting weather data for ({latitude}, {longitude}) from {start_date}")
            
            # Create weather request object
            from models.weather_model import WeatherRequest
            weather_request = WeatherRequest(
                latitude=latitude,
                longitude=longitude,
                disaster_date=end_date,  # Use end_date as disaster_date
                days_before=59  # Use 59 days as per model training
            )
            
            # Fetch weather data using weather service
            weather_success, weather_result = self.weather_service.fetch_weather_data(weather_request)
            
            if weather_success:
                weather_data = weather_result.get('weather_data', {})
                
                # Check if we have the required weather variables
                # Note: Check for original _perc suffix fields before mapping
                # The field mappings will be applied later before prediction
                required_variables_unmapped = [
                    'temperature_C', 'humidity_perc', 'wind_speed_mps', 'precipitation_mm',
                    'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C', 'temperature_min_C',
                    'specific_humidity_g_kg', 'dew_point_C', 'wind_speed_10m_mps', 'cloud_amount_perc',
                    'sea_level_pressure_hPa', 'surface_soil_wetness_perc', 'wind_direction_10m_degrees',
                    'evapotranspiration_wm2', 'root_zone_soil_moisture_perc'
                ]
                missing_vars = [var for var in required_variables_unmapped if var not in weather_data]
                
                if missing_vars:
                    logger.warning(f"Missing weather variables: {missing_vars}")
                
                return True, "Weather data collection successful", weather_data
            else:
                self.service_stats['weather_fetch_failures'] += 1
                error_msg = weather_result.get('error', 'Unknown weather fetch error')
                return False, f"Weather data collection failed: {error_msg}", None
                
        except Exception as e:
            self.service_stats['weather_fetch_failures'] += 1
            logger.error(f"Weather data collection error: {e}")
            return False, f"Weather collection error: {str(e)}", None
    
    def collect_feature_data(self, weather_data: Dict[str, List[float]]) -> Tuple[bool, str, Optional[Dict[str, List[float]]]]:
        """
        Engineer features from weather data
        
        Args:
            weather_data: Raw weather time series data
        
        Returns:
            Tuple of (success, message, feature_data)
        """
        try:
            logger.debug("Engineering features from weather data")
            
            # Create feature request object  
            # Use weather service format directly
            feature_success, feature_result = self.feature_service.process_weather_features(
                weather_data=weather_data,
                event_duration=1.0,
                include_metadata=True
            )
            
            if feature_success:
                feature_data = feature_result.get('engineered_features', {})
                
                # Check if we have the required engineered features
                # Note: Using _perc suffix here as returned by feature service
                required_features = self.prediction_model.ARRAY_FEATURE_COLUMNS[17:]  # Last 19 are engineered
                missing_features = [feat for feat in required_features if feat not in feature_data]
                
                if missing_features:
                    logger.warning(f"Missing engineered features: {missing_features}")
                
                return True, "Feature engineering successful", feature_data
            else:
                self.service_stats['feature_engineering_failures'] += 1
                error_msg = feature_result.get('error', 'Unknown feature engineering error')
                return False, f"Feature engineering failed: {error_msg}", None
                
        except Exception as e:
            self.service_stats['feature_engineering_failures'] += 1
            logger.error(f"Feature engineering error: {e}")
            return False, f"Feature engineering error: {str(e)}", None
    
    def collect_raster_data(self, latitude: float, longitude: float) -> Tuple[bool, str, Optional[Dict[str, float]]]:
        """
        Collect raster data for the location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        
        Returns:
            Tuple of (success, message, raster_data)
        """
        try:
            logger.debug(f"Collecting raster data for ({latitude}, {longitude})")
            
            # Create coordinate list for raster service
            coordinates = [{'latitude': latitude, 'longitude': longitude}]
            
            # Collect raster data
            raster_result = self.raster_service.process_raster_extraction(
                coordinates=coordinates
            )
            
            if raster_result.get('success', False):
                # Get the first (and only) result from data list
                data_list = raster_result.get('data', [])
                if len(data_list) > 0:
                    coord_data = data_list[0]
                    
                    # Extract raster features (exclude latitude/longitude)
                    raster_data = {k: v for k, v in coord_data.items() 
                                 if k not in ['latitude', 'longitude']}
                    
                    # Check if we have the required raster variables
                    required_vars = self.prediction_model.SCALAR_FEATURE_COLUMNS
                    missing_vars = [var for var in required_vars if var not in raster_data]
                    
                    if missing_vars:
                        logger.warning(f"Missing raster variables: {missing_vars}")
                    
                    return True, "Raster data collection successful", raster_data
                else:
                    self.service_stats['raster_fetch_failures'] += 1 
                    return False, "No raster data returned", None
            else:
                self.service_stats['raster_fetch_failures'] += 1
                error_msg = raster_result.get('error', 'Unknown raster fetch error')
                return False, f"Raster data collection failed: {error_msg}", None
                
        except Exception as e:
            self.service_stats['raster_fetch_failures'] += 1
            logger.error(f"Raster data collection error: {e}")
            return False, f"Raster collection error: {str(e)}", None
    
    def predict_disaster_for_location(self, latitude: float, longitude: float, 
                                    reference_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Main prediction method: collect all data and predict disaster risk for location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            reference_date: Reference date for weather data collection (YYYY-MM-DD)
                          If None, uses current date - 60 days for historical analysis
        
        Returns:
            Comprehensive prediction results
        """
        start_time = datetime.now()
        
        try:
            self.service_stats['total_requests'] += 1
            
            # Check if service is initialized
            if not self.service_stats['model_loaded']:
                return {
                    'success': False,
                    'error': 'Service not initialized. Model not loaded.',
                    'prediction': None,
                    'data_collection': {
                        'weather': {'success': False, 'error': 'Service not ready'},
                        'features': {'success': False, 'error': 'Service not ready'},
                        'raster': {'success': False, 'error': 'Service not ready'}
                    },
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Validate coordinates
            coord_valid, coord_message = self.validate_coordinates(latitude, longitude)
            if not coord_valid:
                return {
                    'success': False,
                    'error': f'Invalid coordinates: {coord_message}',
                    'prediction': None,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            logger.info(f"[PREDICTION] HazardGuard prediction for location ({latitude}, {longitude})")
            
            # Calculate date range for weather data
            if reference_date:
                try:
                    ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
                except ValueError:
                    return {
                        'success': False,
                        'error': 'Invalid reference_date format. Use YYYY-MM-DD.',
                        'prediction': None,
                        'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                    }
            else:
                # Keep end_date at least 7 days behind "now" because WeatherRequest
                # enforces NASA lag safety (disaster_date must not be too recent).
                # With a 60-day window (ref_date + 59), this means ref_date = now - 66 days.
                ref_date = datetime.now() - timedelta(days=66)
            
            start_date = ref_date.strftime('%Y-%m-%d')
            end_date = (ref_date + timedelta(days=59)).strftime('%Y-%m-%d')  # 60 days total
            
            logger.info(f"   [DATE_RANGE] Using weather data from {start_date} to {end_date}")
            
            # Data collection tracking
            collection_results = {
                'weather': {'success': False, 'message': '', 'data': None},
                'features': {'success': False, 'message': '', 'data': None},
                'raster': {'success': False, 'message': '', 'data': None}
            }
            
            # Step 1: Collect weather data
            logger.info("   [WEATHER] Collecting weather data...")
            weather_success, weather_message, weather_data = self.collect_weather_data(
                latitude, longitude, start_date, end_date
            )
            collection_results['weather'] = {
                'success': weather_success,
                'message': weather_message,
                'data_points': len(weather_data) if weather_data else 0
            }
            
            if not weather_success:
                self.service_stats['data_collection_failures'] += 1
                logger.error(f"   [WEATHER_ERROR] Failed: {weather_message}")
                logger.error(f"   [WEATHER_ERROR] Dates: {start_date} to {end_date}")
                return {
                    'success': False,
                    'error': f'Weather data collection failed: {weather_message}',
                    'prediction': None,
                    'data_collection': collection_results,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 2: Engineer features
            logger.info("   [FEATURES] Engineering features...")
            feature_success, feature_message, feature_data = self.collect_feature_data(weather_data)
            collection_results['features'] = {
                'success': feature_success,
                'message': feature_message,
                'features_count': len(feature_data) if feature_data else 0
            }
            
            if not feature_success:
                self.service_stats['data_collection_failures'] += 1
                return {
                    'success': False,
                    'error': f'Feature engineering failed: {feature_message}',
                    'prediction': None,
                    'data_collection': collection_results,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 3: Collect raster data
            logger.info("   [RASTER] Collecting raster data...")
            raster_success, raster_message, raster_data = self.collect_raster_data(latitude, longitude)
            collection_results['raster'] = {
                'success': raster_success,
                'message': raster_message,
                'variables_count': len(raster_data) if raster_data else 0
            }
            
            if not raster_success:
                self.service_stats['data_collection_failures'] += 1
                return {
                    'success': False,
                    'error': f'Raster data collection failed: {raster_message}',
                    'prediction': None,
                    'data_collection': collection_results,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 4: Apply field mappings for model compatibility
            # Model expects _% suffix, but services return _perc suffix
            mapped_weather_data = self._apply_field_mappings(weather_data)
            mapped_feature_data = self._apply_field_mappings(feature_data)
            
            # Step 5: Make prediction with metadata for logging
            logger.info("   [PREDICT] Making disaster prediction...")
            
            # Store metadata for debugging logs
            prediction_metadata = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date,
                'end_date': end_date,
                'reference_date': reference_date,
                'days_used': 59
            }
            
            # Pass metadata to prediction model
            if hasattr(self.prediction_model, 'prediction_metadata'):
                self.prediction_model.prediction_metadata = prediction_metadata
            
            prediction_result = self.prediction_model.predict_disaster(
                weather_data=mapped_weather_data,
                feature_data=mapped_feature_data,
                raster_data=raster_data
            )
            
            # Check if prediction was successful
            if prediction_result is None:
                self.service_stats['failed_predictions'] += 1
                return {
                    'success': False,
                    'error': 'Prediction returned None',
                    'prediction': None,
                    'data_collection': collection_results,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # NEW: If disaster is predicted, run disaster type classification
            disaster_types_result = None
            if prediction_result.get('success') and prediction_result.get('prediction') == 'Disaster':
                logger.info("   [DISASTER_DETECTED] Running disaster type classification...")
                try:
                    disaster_types_result = self.disaster_type_classifier.predict_disaster_types(
                        weather_data=mapped_weather_data,
                        feature_data=mapped_feature_data,
                        raster_data=raster_data,
                        lat=latitude,
                        lon=longitude,
                        reference_date=reference_date or start_date
                    )
                    logger.info(f"   [DISASTER_TYPES] Detected: {disaster_types_result['disaster_types']}")
                except Exception as type_error:
                    logger.error(f"   [DISASTER_TYPE_ERROR] Failed to classify disaster types: {type_error}")
                    disaster_types_result = {
                        'disaster_types': [],
                        'probabilities': {},
                        'confidence': 'unknown',
                        'error': str(type_error)
                    }
            
            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update service statistics
            if prediction_result['success']:
                self.service_stats['successful_predictions'] += 1
                
                # Update average processing time
                total_successful = self.service_stats['successful_predictions']
                current_avg = self.service_stats['average_processing_time']
                self.service_stats['average_processing_time'] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
            else:
                self.service_stats['failed_predictions'] += 1
            
            # Format comprehensive response
            response = {
                'success': prediction_result['success'],
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'coordinates_message': coord_message
                },
                'data_collection': collection_results,
                'prediction': prediction_result if prediction_result['success'] else None,
                'disaster_types': disaster_types_result if disaster_types_result else None,  # NEW: Add disaster types
                'processing_info': {
                    'total_processing_time_seconds': processing_time,
                    'weather_date_range': f"{start_date} to {end_date}",
                    'forecast_horizon_days': self.prediction_model.HORIZON,
                    'data_sources': ['NASA_POWER_Weather', 'Engineered_Features', 'Raster_Geographic']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            if not prediction_result['success']:
                response['error'] = prediction_result.get('error', 'Prediction failed')
            else:
                # Only log success details if prediction was successful
                prediction_text = prediction_result.get('prediction', 'Unknown')
                prob_disaster = prediction_result.get('probability', {}).get('disaster', 0)
                logger.info(f"   [SUCCESS] Prediction complete: {prediction_text} (disaster prob: {prob_disaster:.4f})")
                
                # Log disaster types if available
                if disaster_types_result and disaster_types_result.get('disaster_types'):
                    disaster_list = ', '.join(disaster_types_result['disaster_types'])
                    logger.info(f"   [SUCCESS] Disaster types: {disaster_list}")
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.service_stats['failed_predictions'] += 1
            
            logger.error(f"HazardGuard prediction error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': f"Service error: {str(e)}",
                'prediction': None,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            # Calculate uptime
            service_start = datetime.fromisoformat(self.service_stats['service_start_time'])
            uptime_seconds = (datetime.now() - service_start).total_seconds()
            
            # Get model info
            model_info = self.prediction_model.get_model_info()
            
            # Calculate success rates
            total_requests = self.service_stats['total_requests']
            success_rate = (self.service_stats['successful_predictions'] / total_requests * 100) if total_requests > 0 else 0

            weather_provider_details = {
                'provider': 'unknown',
                'fallback_provider': None,
                'last_provider': None,
                'last_error': ''
            }
            if hasattr(self.weather_service, 'get_service_status'):
                weather_status = self.weather_service.get_service_status() or {}
                weather_provider_details = {
                    'provider': weather_status.get('service', 'unknown'),
                    'fallback_provider': weather_status.get('fallback_provider'),
                    'last_provider': weather_status.get('last_provider'),
                    'last_error': weather_status.get('last_error', '')
                }
            
            return {
                'service_status': 'ready' if self.service_stats['model_loaded'] else 'not_initialized',
                'uptime_seconds': uptime_seconds,
                'uptime_hours': uptime_seconds / 3600,
                'model_loaded': self.service_stats['model_loaded'],
                'model_info': model_info,
                'statistics': {
                    'total_requests': total_requests,
                    'successful_predictions': self.service_stats['successful_predictions'],
                    'failed_predictions': self.service_stats['failed_predictions'],
                    'success_rate_percent': success_rate,
                    'data_collection_failures': self.service_stats['data_collection_failures'],
                    'weather_fetch_failures': self.service_stats['weather_fetch_failures'],
                    'feature_engineering_failures': self.service_stats['feature_engineering_failures'],
                    'raster_fetch_failures': self.service_stats['raster_fetch_failures'],
                    'average_processing_time_seconds': self.service_stats['average_processing_time']
                },
                'service_dependencies': {
                    'weather_service': weather_provider_details,
                    'feature_service': 'FeatureEngineering',
                    'raster_service': 'RasterData',
                    'prediction_model': 'XGBoost_Binary_Classifier'
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                'service_status': 'error',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def reset_statistics(self) -> Dict[str, str]:
        """Reset service statistics"""
        try:
            # Reset service stats
            self.service_stats.update({
                'total_requests': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'data_collection_failures': 0,
                'weather_fetch_failures': 0,
                'feature_engineering_failures': 0,
                'raster_fetch_failures': 0,
                'average_processing_time': 0.0
            })
            
            # Reset model statistics
            self.prediction_model.reset_statistics()
            
            logger.info("HazardGuard service statistics reset")
            
            return {
                'status': 'success',
                'message': 'All statistics reset successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error resetting statistics: {e}")
            return {
                'status': 'error',
                'message': f"Failed to reset statistics: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }