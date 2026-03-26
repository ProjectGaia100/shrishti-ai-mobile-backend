"""
Post-Disaster Feature Engineering Service for HazardGuard System
Business logic layer for post-disaster feature engineering operations
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json
import traceback

from models.post_disaster_feature_engineering_model import PostDisasterFeatureEngineeringModel

logger = logging.getLogger(__name__)

class PostDisasterFeatureEngineeringService:
    """Service class for post-disaster feature engineering operations"""
    
    def __init__(self):
        """Initialize the post-disaster feature engineering service"""
        self.model = PostDisasterFeatureEngineeringModel(days_count=60)
        self.service_stats = {
            'service_start_time': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_coordinates_processed': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("PostDisasterFeatureEngineeringService initialized")
    
    def validate_coordinates(self, coordinates: Any) -> Tuple[bool, str, Optional[List[List[float]]]]:
        """
        Validate coordinate input format
        
        Args:
            coordinates: Input coordinates data
        
        Returns:
            Tuple of (is_valid, message, parsed_coordinates)
        """
        try:
            if not coordinates:
                return False, "No coordinates provided", None
            
            # Handle string input (JSON)
            if isinstance(coordinates, str):
                try:
                    coordinates = json.loads(coordinates)
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON format: {str(e)}", None
            
            # Ensure it's a list
            if not isinstance(coordinates, list):
                return False, "Coordinates must be a list", None
            
            # Validate each coordinate pair
            parsed_coordinates = []
            for i, coord in enumerate(coordinates):
                if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                    return False, f"Coordinate {i+1} must be [lat, lon] format", None
                
                try:
                    lat, lon = float(coord[0]), float(coord[1])
                    
                    # Validate latitude and longitude ranges
                    if not (-90 <= lat <= 90):
                        return False, f"Invalid latitude {lat} in coordinate {i+1} (must be -90 to 90)", None
                    if not (-180 <= lon <= 180):
                        return False, f"Invalid longitude {lon} in coordinate {i+1} (must be -180 to 180)", None
                    
                    parsed_coordinates.append([lat, lon])
                    
                except (ValueError, TypeError):
                    return False, f"Coordinate {i+1} contains non-numeric values", None
            
            if len(parsed_coordinates) == 0:
                return False, "No valid coordinates found", None
            
            return True, f"Validated {len(parsed_coordinates)} coordinates", parsed_coordinates
            
        except Exception as e:
            logger.error(f"Coordinate validation error: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def validate_weather_data(self, weather_data: Any) -> Tuple[bool, str, Optional[Dict[str, List[float]]]]:
        """
        Validate weather data input format
        
        Args:
            weather_data: Input weather data
        
        Returns:
            Tuple of (is_valid, message, parsed_weather_data)
        """
        try:
            if not weather_data:
                return False, "No weather data provided", None
            
            # Handle string input (JSON)
            if isinstance(weather_data, str):
                try:
                    weather_data = json.loads(weather_data)
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON format: {str(e)}", None
            
            # Ensure it's a dictionary
            if not isinstance(weather_data, dict):
                return False, "Weather data must be a dictionary", None
            
            # Validate using model validation
            is_valid, validation_message = self.model.validate_weather_data(weather_data)
            if not is_valid:
                return False, validation_message, None
            
            return True, "Weather data validation successful", weather_data
            
        except Exception as e:
            logger.error(f"Weather data validation error: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def process_single_coordinate_features(self, weather_data: Dict[str, List[float]], 
                                         coordinate: Optional[List[float]] = None,
                                         global_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Process post-disaster feature engineering for a single coordinate
        
        Args:
            weather_data: Weather time series data
            coordinate: Optional coordinate for metadata
            global_stats: Optional global statistics for normalization
        
        Returns:
            Result dictionary with features and metadata
        """
        start_time = datetime.now()
        
        try:
            self.service_stats['total_requests'] += 1
            
            # Validate weather data
            is_valid, validation_message, validated_weather = self.validate_weather_data(weather_data)
            if not is_valid:
                self.service_stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': f"Weather data validation failed: {validation_message}",
                    'coordinate': coordinate,
                    'features': None,
                    'processing_time_seconds': 0.0
                }
            
            # Engineer features using model
            result = self.model.engineer_single_coordinate_features(validated_weather, global_stats)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update service statistics
            if result['success']:
                self.service_stats['successful_requests'] += 1
                self.service_stats['total_coordinates_processed'] += 1
                
                # Update average processing time
                total_successful = self.service_stats['successful_requests']
                current_avg = self.service_stats['average_processing_time']
                self.service_stats['average_processing_time'] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
            else:
                self.service_stats['failed_requests'] += 1
            
            # Format response
            response = {
                'success': result['success'],
                'coordinate': coordinate,
                'features': result.get('features'),
                'metadata': result.get('metadata', {}),
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if not result['success']:
                response['error'] = result.get('error', 'Unknown error')
            
            logger.info(f"Single coordinate processing {'successful' if result['success'] else 'failed'}: {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.service_stats['failed_requests'] += 1
            
            logger.error(f"Service error in single coordinate processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': f"Service error: {str(e)}",
                'coordinate': coordinate,
                'features': None,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def process_batch_features(self, weather_datasets: List[Dict[str, List[float]]], 
                             coordinates: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Process post-disaster feature engineering for multiple coordinates
        
        Args:
            weather_datasets: List of weather data dictionaries
            coordinates: Optional list of coordinates for metadata
        
        Returns:
            Batch processing results with features and statistics
        """
        start_time = datetime.now()
        
        try:
            self.service_stats['total_requests'] += 1
            
            # Validate inputs
            if not weather_datasets:
                self.service_stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': "No weather datasets provided",
                    'results': [],
                    'processing_time_seconds': 0.0
                }
            
            # Validate coordinates if provided
            if coordinates:
                if len(coordinates) != len(weather_datasets):
                    self.service_stats['failed_requests'] += 1
                    return {
                        'success': False,
                        'error': f"Coordinates count ({len(coordinates)}) doesn't match weather datasets count ({len(weather_datasets)})",
                        'results': [],
                        'processing_time_seconds': 0.0
                    }
            
            logger.info(f"Processing batch features for {len(weather_datasets)} coordinates")
            
            # Process using model (with global statistics)
            model_results = self.model.engineer_batch_features(weather_datasets)
            
            # Format results with coordinate information
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, model_result in enumerate(model_results):
                coordinate = coordinates[i] if coordinates else None
                
                result = {
                    'coordinate_index': i + 1,
                    'coordinate': coordinate,
                    'success': model_result['success'],
                    'features': model_result.get('features'),
                    'metadata': model_result.get('metadata', {})
                }
                
                if not model_result['success']:
                    result['error'] = model_result.get('error', 'Unknown error')
                    failed_count += 1
                else:
                    successful_count += 1
                
                results.append(result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update service statistics
            if successful_count > 0:
                self.service_stats['successful_requests'] += 1
                self.service_stats['total_coordinates_processed'] += successful_count
                
                # Update average processing time
                total_successful = self.service_stats['successful_requests']
                current_avg = self.service_stats['average_processing_time']
                self.service_stats['average_processing_time'] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
            
            if failed_count > 0:
                self.service_stats['failed_requests'] += failed_count
            
            # Prepare response
            response = {
                'success': successful_count > 0,
                'total_coordinates': len(weather_datasets),
                'successful_coordinates': successful_count,
                'failed_coordinates': failed_count,
                'success_rate_percent': (successful_count / len(weather_datasets) * 100) if weather_datasets else 0,
                'results': results,
                'global_statistics': self.model.global_stats,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Batch processing completed: {successful_count}/{len(weather_datasets)} successful ({processing_time:.3f}s)")
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.service_stats['failed_requests'] += 1
            
            logger.error(f"Service error in batch processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': f"Batch processing error: {str(e)}",
                'results': [],
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def export_to_dataframe(self, results: List[Dict[str, Any]], 
                           include_metadata: bool = True) -> Optional[pd.DataFrame]:
        """
        Export feature engineering results to pandas DataFrame
        
        Args:
            results: List of feature engineering results
            include_metadata: Whether to include coordinate and metadata columns
        
        Returns:
            pandas DataFrame or None if export fails
        """
        try:
            if not results:
                logger.warning("No results provided for DataFrame export")
                return None
            
            export_data = []
            
            for result in results:
                if not result.get('success', False):
                    continue
                
                features = result.get('features', {})
                if not features:
                    continue
                
                # Get the number of days from the first feature
                first_feature = next(iter(features.values()), [])
                days_count = len(first_feature)
                
                # Create rows for each day
                for day in range(days_count):
                    row_data = {}
                    
                    # Add metadata columns if requested
                    if include_metadata:
                        row_data['coordinate_index'] = result.get('coordinate_index', 0)
                        coordinate = result.get('coordinate', [None, None])
                        row_data['latitude'] = coordinate[0] if len(coordinate) >= 1 else None
                        row_data['longitude'] = coordinate[1] if len(coordinate) >= 2 else None
                        row_data['day'] = day + 1
                    
                    # Add feature values for this day
                    for feature_name, feature_values in features.items():
                        if day < len(feature_values):
                            value = feature_values[day]
                            # Handle numpy NaN and convert to pandas-compatible NaN
                            if pd.isna(value):
                                row_data[feature_name] = np.nan
                            else:
                                row_data[feature_name] = float(value)
                        else:
                            row_data[feature_name] = np.nan
                    
                    export_data.append(row_data)
            
            if not export_data:
                logger.warning("No valid data to export to DataFrame")
                return None
            
            df = pd.DataFrame(export_data)
            logger.info(f"DataFrame export successful: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame export error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_feature_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Get descriptions of all engineered features"""
        return self.model.get_feature_descriptions()
    
    def get_input_variables(self) -> List[str]:
        """Get list of required input variables"""
        return self.model.POST_WEATHER_VARIABLES.copy()
    
    def get_output_variables(self) -> List[str]:
        """Get list of output feature variables"""
        return self.model.POST_FEATURE_VARIABLES.copy()
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get service health and performance statistics
        
        Returns:
            Service health information
        """
        try:
            # Get model statistics
            model_stats = self.model.get_processing_statistics()
            
            # Calculate service uptime
            service_start = datetime.fromisoformat(self.service_stats['service_start_time'])
            uptime_seconds = (datetime.now() - service_start).total_seconds()
            
            total_requests = self.service_stats['total_requests']
            
            return {
                'service_status': 'healthy',
                'service_uptime_seconds': uptime_seconds,
                'service_uptime_hours': uptime_seconds / 3600,
                'total_requests': total_requests,
                'successful_requests': self.service_stats['successful_requests'],
                'failed_requests': self.service_stats['failed_requests'],
                'success_rate_percent': (self.service_stats['successful_requests'] / total_requests * 100) if total_requests > 0 else 0,
                'total_coordinates_processed': self.service_stats['total_coordinates_processed'],
                'average_processing_time_seconds': self.service_stats['average_processing_time'],
                'model_statistics': model_stats,
                'feature_counts': {
                    'input_variables': len(self.model.POST_WEATHER_VARIABLES),
                    'output_variables': len(self.model.POST_FEATURE_VARIABLES),
                    'days_per_coordinate': self.model.days_count
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return {
                'service_status': 'error',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def reset_statistics(self) -> Dict[str, str]:
        """Reset service and model statistics"""
        try:
            # Reset service statistics
            self.service_stats = {
                'service_start_time': datetime.now().isoformat(),
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_coordinates_processed': 0,
                'average_processing_time': 0.0
            }
            
            # Reset model statistics
            self.model.processing_stats = {
                'total_processed': 0,
                'successful_calculations': 0,
                'failed_calculations': 0,
                'nan_count': 0
            }
            
            logger.info("Service and model statistics reset")
            
            return {
                'status': 'success',
                'message': 'Statistics reset successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error resetting statistics: {e}")
            return {
                'status': 'error',
                'message': f"Failed to reset statistics: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }