"""
Feature Engineering Service
Handles weather feature engineering processing and data transformation
"""
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from models.feature_engineering_model import WeatherFeatureModel

class FeatureEngineeringService:
    """Service for weather feature engineering operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialized = True
        self.processing_stats = {
            'total_processed': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'total_processing_time': 0.0
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and configuration"""
        return {
            'service': 'Feature Engineering Service',
            'initialized': self.initialized,
            'supported_features': len(WeatherFeatureModel.ENGINEERED_FEATURES),
            'required_weather_fields': len(WeatherFeatureModel.WEATHER_FIELDS),
            'processing_stats': self.processing_stats.copy(),
            'feature_descriptions': WeatherFeatureModel.FEATURE_DESCRIPTIONS
        }
    
    def validate_input_data(self, weather_data: Dict[str, List]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input weather data for feature engineering
        
        Args:
            weather_data: Dictionary with weather field lists
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        try:
            validation = WeatherFeatureModel.validate_weather_data(weather_data)
            
            if validation['valid']:
                self.logger.info(f"Weather data validation passed: {validation['field_count']} fields, "
                               f"{validation['days_count']} days")
                
                # Additional checks for feature engineering readiness
                if validation['days_count'] < 1:
                    validation['valid'] = False
                    validation['errors'].append("At least 1 day of data required")
                
                if validation['field_count'] < len(WeatherFeatureModel.WEATHER_FIELDS):
                    validation['warnings'].append(
                        f"Only {validation['field_count']}/{len(WeatherFeatureModel.WEATHER_FIELDS)} "
                        "weather fields provided - some features may be computed as NaN"
                    )
            else:
                self.logger.warning(f"Weather data validation failed: {validation['errors']}")
            
            return validation['valid'], validation
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False, {
                'valid': False,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': [],
                'days_count': 0,
                'field_count': 0
            }
    
    def process_weather_features(self, weather_data: Dict[str, List], 
                               event_duration: Optional[float] = None,
                               include_metadata: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Process weather data and compute engineered features
        
        Args:
            weather_data: Dictionary with weather field lists (60 days each)
            event_duration: Duration of event in days (optional, defaults to 1.0)
            include_metadata: Whether to include processing metadata
            
        Returns:
            Tuple of (success, result_data)
        """
        start_time = time.time()
        
        try:
            # Validate input data
            is_valid, validation = self.validate_input_data(weather_data)
            if not is_valid:
                self.processing_stats['failed_processes'] += 1
                return False, {
                    'error': 'Input validation failed',
                    'validation': validation,
                    'status': 'validation_error'
                }
            
            # Set default event duration
            if event_duration is None:
                event_duration = 1.0
            elif event_duration <= 0:
                event_duration = 1.0
                validation['warnings'].append("Invalid event duration, using default 1.0 days")
            
            self.logger.info(f"Processing feature engineering for {validation['days_count']} days, "
                           f"event duration: {event_duration} days")
            
            # Compute engineered features
            engineered_features = WeatherFeatureModel.compute_engineered_features(
                weather_data, event_duration
            )
            
            # Compute processing statistics
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['successful_processes'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            # Build result
            result = {
                'engineered_features': engineered_features,
                'validation': validation,
                'status': 'success'
            }
            
            if include_metadata:
                result['metadata'] = {
                    'days_processed': validation['days_count'],
                    'features_computed': len(engineered_features),
                    'event_duration': event_duration,
                    'processing_time': processing_time,
                    'feature_names': list(engineered_features.keys()),
                    'original_fields': list(weather_data.keys())
                }
            
            self.logger.info(f"Feature engineering completed successfully: "
                           f"{len(engineered_features)} features, "
                           f"{processing_time:.3f}s processing time")
            
            return True, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['failed_processes'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            self.logger.error(f"Feature engineering failed: {str(e)}")
            return False, {
                'error': f'Feature engineering failed: {str(e)}',
                'processing_time': processing_time,
                'status': 'processing_error'
            }
    
    def process_batch_features(self, batch_data: List[Dict[str, Any]], 
                             include_metadata: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Process multiple weather datasets for feature engineering
        
        Args:
            batch_data: List of weather data dictionaries with optional event_duration
            include_metadata: Whether to include processing metadata
            
        Returns:
            Tuple of (success, batch_results)
        """
        start_time = time.time()
        
        try:
            if not batch_data or len(batch_data) > 100:  # Limit batch size
                return False, {
                    'error': f'Invalid batch size: {len(batch_data)} (max 100 allowed)',
                    'status': 'validation_error'
                }
            
            self.logger.info(f"Processing batch feature engineering for {len(batch_data)} datasets")
            
            batch_results = []
            successful_count = 0
            failed_count = 0
            
            for i, data_item in enumerate(batch_data):
                # Extract weather data and event duration
                weather_data = data_item.get('weather_data', {})
                event_duration = data_item.get('event_duration', 1.0)
                item_id = data_item.get('id', f'item_{i}')
                
                # Process this item
                success, result = self.process_weather_features(
                    weather_data, event_duration, include_metadata=False
                )
                
                # Add to batch results
                batch_results.append({
                    'id': item_id,
                    'success': success,
                    'result': result
                })
                
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
            
            processing_time = time.time() - start_time
            
            # Build batch response
            result = {
                'batch_results': batch_results,
                'summary': {
                    'total_items': len(batch_data),
                    'successful': successful_count,
                    'failed': failed_count,
                    'success_rate': successful_count / len(batch_data) * 100
                },
                'status': 'success'
            }
            
            if include_metadata:
                result['metadata'] = {
                    'batch_processing_time': processing_time,
                    'average_time_per_item': processing_time / len(batch_data),
                    'batch_size': len(batch_data)
                }
            
            self.logger.info(f"Batch processing completed: {successful_count}/{len(batch_data)} successful, "
                           f"{processing_time:.3f}s total time")
            
            return True, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Batch feature engineering failed: {str(e)}")
            return False, {
                'error': f'Batch processing failed: {str(e)}',
                'processing_time': processing_time,
                'status': 'batch_error'
            }
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available engineered features"""
        return {
            'engineered_features': {
                name: WeatherFeatureModel.FEATURE_DESCRIPTIONS.get(name, 'No description')
                for name in WeatherFeatureModel.ENGINEERED_FEATURES
            },
            'required_weather_fields': {
                field: f"Weather field: {field}"
                for field in WeatherFeatureModel.WEATHER_FIELDS
            },
            'feature_count': len(WeatherFeatureModel.ENGINEERED_FEATURES),
            'weather_field_count': len(WeatherFeatureModel.WEATHER_FIELDS),
            'nan_handling': 'NaN values in input produce NaN values in output for that specific day only'
        }
    
    def create_feature_dataframe(self, weather_data: Dict[str, List], 
                               engineered_features: Dict[str, List],
                               disaster_date: str, days_before: int) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with weather data and engineered features
        
        Args:
            weather_data: Original weather data
            engineered_features: Computed engineered features
            disaster_date: Disaster date string
            days_before: Number of days before disaster
            
        Returns:
            Combined DataFrame with dates, weather data, and features
        """
        try:
            # Import pandas for DataFrame creation
            import pandas as pd
            from datetime import datetime, timedelta
            
            # Calculate date range
            disaster_dt = datetime.strptime(disaster_date, '%Y-%m-%d')
            end_date = disaster_dt
            start_date = end_date - timedelta(days=days_before - 1)
            
            # Generate date range
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_range.append(current_date)
                current_date += timedelta(days=1)
            
            date_strings = [dt.strftime('%Y-%m-%d') for dt in date_range]
            
            # Create DataFrame starting with dates
            df = pd.DataFrame({'date': date_strings})
            
            # Add original weather data
            for field, values in weather_data.items():
                df[field] = values[:len(date_range)]
            
            # Add engineered features
            for feature, values in engineered_features.items():
                df[feature] = values[:len(date_range)]
            
            return df
            
        except ImportError:
            self.logger.warning("Pandas not available for DataFrame creation")
            raise
        except Exception as e:
            self.logger.error(f"DataFrame creation failed: {str(e)}")
            raise
    
    def process_and_export(self, weather_data: Dict[str, List],
                         disaster_date: str, days_before: int,
                         event_duration: Optional[float] = None,
                         export_format: str = 'dict') -> Tuple[bool, Dict[str, Any]]:
        """
        Process features and export in specified format
        
        Args:
            weather_data: Weather data dictionary
            disaster_date: Disaster date string
            days_before: Number of days before disaster 
            event_duration: Event duration in days
            export_format: 'dict', 'dataframe', or 'json'
            
        Returns:
            Tuple of (success, exported_data)
        """
        try:
            # Process engineered features
            success, result = self.process_weather_features(
                weather_data, event_duration, include_metadata=True
            )
            
            if not success:
                return False, result
            
            engineered_features = result['engineered_features']
            
            # Export in requested format
            if export_format == 'dataframe':
                try:
                    df = self.create_feature_dataframe(
                        weather_data, engineered_features, disaster_date, days_before
                    )
                    result['export'] = {
                        'format': 'dataframe',
                        'dataframe': df,
                        'shape': df.shape,
                        'columns': list(df.columns)
                    }
                except Exception as e:
                    return False, {
                        'error': f'DataFrame export failed: {str(e)}',
                        'status': 'export_error'
                    }
            
            elif export_format == 'json':
                # Convert to JSON-serializable format
                json_data = {}
                for feature, values in engineered_features.items():
                    # Convert NaN to None for JSON compatibility
                    json_values = [None if (isinstance(v, float) and np.isnan(v)) else v for v in values]
                    json_data[feature] = json_values
                
                result['export'] = {
                    'format': 'json',
                    'features': json_data,
                    'feature_count': len(json_data),
                    'days_count': len(next(iter(json_data.values())))
                }
            
            else:  # Default to dict format
                result['export'] = {
                    'format': 'dict',
                    'features': engineered_features,
                    'feature_count': len(engineered_features),
                    'days_count': len(next(iter(engineered_features.values())))
                }
            
            return True, result
            
        except Exception as e:
            self.logger.error(f"Process and export failed: {str(e)}")
            return False, {
                'error': f'Process and export failed: {str(e)}',
                'status': 'export_error'
            }