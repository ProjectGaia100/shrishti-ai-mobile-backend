"""
WeatherWise Prediction Service
Orchestrates data collection and LSTM weather forecasting for different disaster contexts
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from models.weatherwise_prediction_model import WeatherWisePredictionModel
from services.weather_service import NASAPowerService
from services.feature_engineering_service import FeatureEngineeringService

logger = logging.getLogger(__name__)

class WeatherWisePredictionService:
    """Service for orchestrating WeatherWise weather forecasting"""
    
    def __init__(self, weather_service: Optional[NASAPowerService] = None,
                 feature_service: Optional[FeatureEngineeringService] = None):
        """
        Initialize WeatherWise prediction service
        
        Args:
            weather_service: NASA POWER weather service instance
            feature_service: Feature engineering service instance
        """
        # Initialize LSTM prediction models
        self.prediction_model = WeatherWisePredictionModel()
        
        # Initialize or use provided services (same as HazardGuard, but no raster service)
        self.weather_service = weather_service or NASAPowerService()
        self.feature_service = feature_service or FeatureEngineeringService()
        
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
            'successful_forecasts': 0,
            'failed_forecasts': 0,
            'data_collection_failures': 0,
            'weather_fetch_failures': 0,
            'feature_engineering_failures': 0,
            'average_processing_time': 0.0,
            'models_loaded': False
        }
        
        logger.info("WeatherWise prediction service initialized")
    
    def initialize_service(self) -> Tuple[bool, str]:
        """
        Initialize the service by loading the LSTM weather models
        
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("Initializing WeatherWise prediction service...")
            
            # Load LSTM weather prediction models
            models_loaded = self.prediction_model.load_models()
            
            if models_loaded:
                self.service_stats['models_loaded'] = True
                available_models = self.prediction_model.get_available_models()
                logger.info(f"[SUCCESS] WeatherWise service initialization successful")
                logger.info(f"[SUCCESS] Available models: {available_models}")
                return True, f"Service initialized with {len(available_models)} models"
            else:
                logger.error("[ERROR] Failed to load LSTM models")
                return False, "Failed to load LSTM weather models"
                
        except Exception as e:
            logger.error(f"WeatherWise service initialization error: {e}")
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
        Collect weather data for LSTM forecasting (same as HazardGuard)
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Tuple of (success, message, weather_data)
        """
        try:
            logger.debug(f"Collecting weather data for LSTM forecasting at ({latitude}, {longitude}) from {start_date}")
            
            # Create weather request object
            from models.weather_model import WeatherRequest
            weather_request = WeatherRequest(
                latitude=latitude,
                longitude=longitude,
                disaster_date=end_date,  # Use end_date as disaster_date
                days_before=60  # Use 60 days for LSTM input sequence (model expects 60 timesteps)
            )
            
            # Fetch weather data using weather service
            weather_success, weather_result = self.weather_service.fetch_weather_data(weather_request)
            
            if weather_success:
                weather_data = weather_result.get('weather_data', {})
                
                # Do NOT apply field mappings here - keep _perc suffix for feature engineering
                # Mappings will be applied later when preparing LSTM input
                
                logger.info(f"[WEATHERWISE] Weather data collected: {len(weather_data)} variables")
                return True, "Weather data collection successful", weather_data
            else:
                self.service_stats['weather_fetch_failures'] += 1
                error_msg = weather_result.get('error', 'Unknown weather fetch error')
                return False, f"Weather data collection failed: {error_msg}", None
                
        except Exception as e:
            self.service_stats['weather_fetch_failures'] += 1
            logger.error(f"WeatherWise weather data collection error: {e}")
            return False, f"Weather collection error: {str(e)}", None
    
    def collect_feature_data(self, weather_data: Dict[str, List[float]], 
                           latitude: float, longitude: float, reference_date: str) -> Tuple[bool, str, Optional[Dict[str, List[float]]]]:
        """
        Collect engineered features for LSTM forecasting (same as HazardGuard but no raster)
        
        Args:
            weather_data: Weather data dictionary
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            reference_date: Reference date in YYYY-MM-DD format
        
        Returns:
            Tuple of (success, message, feature_data)
        """
        try:
            logger.debug(f"Engineering features for LSTM forecasting")
            
            # Generate engineered features using the existing feature service
            feature_success, feature_result = self.feature_service.process_weather_features(
                weather_data=weather_data,
                event_duration=1.0,
                include_metadata=True
            )
            
            if feature_success:
                feature_data = feature_result.get('engineered_features', {})
                logger.info(f"[WEATHERWISE] Feature engineering completed: {len(feature_data)} features")
                return True, "Feature engineering successful", feature_data
            else:
                self.service_stats['feature_engineering_failures'] += 1
                error_msg = feature_result.get('error', 'Unknown feature engineering error')
                return False, f"Feature engineering failed: {error_msg}", None
                
        except Exception as e:
            self.service_stats['feature_engineering_failures'] += 1
            logger.error(f"WeatherWise feature engineering error: {e}")
            return False, f"Feature engineering error: {str(e)}", None
    
    def generate_weather_forecast(self, latitude: float, longitude: float, 
                                reference_date: str = None, disaster_type: str = 'Normal',
                                forecast_days: int = 60) -> Dict[str, Any]:
        """
        Generate weather forecast for location using LSTM models
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            reference_date: Reference date in YYYY-MM-DD format (optional)
            disaster_type: Disaster context for model selection
            forecast_days: Number of days to forecast (default 60)
        
        Returns:
            Dict containing forecast results
        """
        start_time = datetime.now()
        self.service_stats['total_requests'] += 1
        
        try:
            logger.info(f"[WEATHERWISE] Starting forecast generation for ({latitude}, {longitude})")
            logger.info(f"[WEATHERWISE] Disaster context: {disaster_type}, Forecast days: {forecast_days}")
            
            # Validate coordinates
            coord_valid, coord_message = self.validate_coordinates(latitude, longitude)
            if not coord_valid:
                return {
                    'success': False,
                    'error': coord_message,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Calculate date range for historical data (60 days before reference date)
            if reference_date:
                end_date = datetime.strptime(reference_date, '%Y-%m-%d')
            else:
                # Default to current date minus 7 days (NASA POWER lag)
                end_date = datetime.now() - timedelta(days=7)
            
            start_date = end_date - timedelta(days=60)  # 60 days before end_date
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"[WEATHERWISE] Historical data range: {start_date_str} to {end_date_str}")
            
            # Collect weather data
            weather_success, weather_message, weather_data = self.collect_weather_data(
                latitude, longitude, start_date_str, end_date_str
            )
            
            if not weather_success:
                self.service_stats['data_collection_failures'] += 1
                return {
                    'success': False,
                    'error': f'Weather data collection failed: {weather_message}',
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Collect engineered features (excluding raster data)
            feature_success, feature_message, feature_data = self.collect_feature_data(
                weather_data, latitude, longitude, end_date_str
            )
            
            if not feature_success:
                self.service_stats['data_collection_failures'] += 1
                return {
                    'success': False,
                    'error': f'Feature engineering failed: {feature_message}',
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds()
                }
            
            # Generate forecast using LSTM model
            logger.info(f"[WEATHERWISE] Generating {forecast_days}-day forecast with {disaster_type} context")
            
            # Apply field mappings for LSTM model (_perc -> _%)
            mapped_weather_data = self._apply_field_mappings(weather_data)
            
            forecast_result = self.prediction_model.predict_weather_forecast(
                weather_data=mapped_weather_data,
                feature_data=feature_data,
                disaster_type=disaster_type,
                forecast_days=forecast_days
            )
            
            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if forecast_result.get('success'):
                self.service_stats['successful_forecasts'] += 1
                
                # Update average processing time
                total_successful = self.service_stats['successful_forecasts']
                current_avg = self.service_stats['average_processing_time']
                self.service_stats['average_processing_time'] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
                
                logger.info(f"[SUCCESS] WeatherWise forecast generated in {processing_time:.2f}s")
                
                # Add service metadata to response
                forecast_result.update({
                    'location': {
                        'latitude': latitude,
                        'longitude': longitude,
                        'coordinates_message': coord_message
                    },
                    'data_collection': {
                        'weather_data_success': weather_success,
                        'feature_engineering_success': feature_success,
                        'historical_data_range': f'{start_date_str} to {end_date_str}'
                    },
                    'processing_info': {
                        'total_processing_time_seconds': processing_time,
                        'data_sources': ['NASA_POWER_Weather', 'Engineered_Features'],
                        'model_context': disaster_type
                    }
                })
                
            else:
                self.service_stats['failed_forecasts'] += 1
                forecast_result['processing_time_seconds'] = processing_time
                
            return forecast_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.service_stats['failed_forecasts'] += 1
            
            logger.error(f"WeatherWise forecast error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': f'Forecast generation error: {str(e)}',
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get WeatherWise service health information"""
        available_models = self.prediction_model.get_available_models() if self.service_stats['models_loaded'] else []
        
        return {
            'service_name': 'WeatherWise',
            'status': 'healthy' if self.service_stats['models_loaded'] else 'unhealthy',
            'models_loaded': self.service_stats['models_loaded'],
            'available_disaster_contexts': available_models,
            'statistics': self.service_stats,
            'supported_forecast_variables': self.prediction_model.forecast_variables if hasattr(self.prediction_model, 'forecast_variables') else [],
            'default_forecast_horizon_days': 60,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available disaster context models"""
        return self.prediction_model.get_available_models()