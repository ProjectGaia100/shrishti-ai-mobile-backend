"""
Post-Disaster Weather Data Service for HazardGuard System
Business logic layer for post-disaster weather data operations
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import os

from models.post_disaster_weather_model import PostDisasterWeatherModel

logger = logging.getLogger(__name__)

class PostDisasterWeatherService:
    """Service layer for post-disaster weather data operations"""
    
    def __init__(self, 
                 days_after_disaster: int = 60,
                 max_workers: int = 1,
                 retry_limit: int = 5,
                 retry_delay: int = 15,
                 rate_limit_pause: int = 900,
                 request_delay: float = 0.5):
        """Initialize post-disaster weather service"""
        try:
            self.model = PostDisasterWeatherModel(
                days_after_disaster=days_after_disaster,
                max_workers=max_workers,
                retry_limit=retry_limit, 
                retry_delay=retry_delay,
                rate_limit_pause=rate_limit_pause,
                request_delay=request_delay
            )
            
            self.processing_stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'coordinates_processed': 0,
                'last_request_time': None,
                'service_status': 'ready'
            }
            
            logger.info("PostDisasterWeatherService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostDisasterWeatherService: {e}")
            raise
    
    def validate_coordinates(self, coordinates: List[Dict[str, float]]) -> Tuple[bool, str]:
        """Validate coordinate input format and ranges"""
        return self.model.validate_coordinates(coordinates)
    
    def validate_disaster_dates(self, disaster_dates: Union[List[str], List[datetime], str, datetime]) -> Tuple[bool, str, List[datetime]]:
        """Validate disaster date inputs"""
        try:
            # Convert single date to list
            if isinstance(disaster_dates, (str, datetime)):
                disaster_dates = [disaster_dates]
            
            if not disaster_dates:
                return False, "Disaster dates cannot be empty", []
            
            parsed_dates = []
            for i, date in enumerate(disaster_dates):
                is_valid, message, parsed_date = self.model.validate_disaster_date(date)
                if not is_valid:
                    return False, f"Date {i}: {message}", []
                parsed_dates.append(parsed_date)
            
            return True, "All dates are valid", parsed_dates
            
        except Exception as e:
            logger.error(f"Date validation error: {e}")
            return False, f"Date validation error: {str(e)}", []
    
    def process_post_disaster_weather(self, 
                                    coordinates: List[Dict[str, float]], 
                                    disaster_dates: Union[List[str], List[datetime], str, datetime],
                                    variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process post-disaster weather data extraction
        
        Args:
            coordinates: List of coordinate dictionaries with 'latitude' and 'longitude' 
            disaster_dates: Disaster dates (single date or list matching coordinates)
            variables: Optional list of specific variables to extract (default: all)
        
        Returns:
            Dictionary with extraction results and metadata
        """
        try:
            self.processing_stats['total_requests'] += 1
            self.processing_stats['last_request_time'] = datetime.now().isoformat()
            
            # Validate coordinates
            is_valid, coord_message = self.validate_coordinates(coordinates)
            if not is_valid:
                self.processing_stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': f"Invalid coordinates: {coord_message}",
                    'data': None
                }
            
            # Validate and parse dates
            is_valid, date_message, parsed_dates = self.validate_disaster_dates(disaster_dates) 
            if not is_valid:
                self.processing_stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': f"Invalid disaster dates: {date_message}",
                    'data': None
                }
            
            # Handle single date for multiple coordinates
            if len(parsed_dates) == 1 and len(coordinates) > 1:
                parsed_dates = parsed_dates * len(coordinates)
            elif len(parsed_dates) != len(coordinates):
                self.processing_stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': f"Mismatch: {len(coordinates)} coordinates but {len(parsed_dates)} dates",
                    'data': None
                }
            
            # Validate variable selection
            if variables:
                available_vars = list(self.model.WEATHER_FIELDS.values())
                invalid_vars = [v for v in variables if v not in available_vars]
                if invalid_vars:
                    return {
                        'success': False,
                        'error': f"Invalid variables: {invalid_vars}. Available: {available_vars}",
                        'data': None
                    }
            
            logger.info(f"Processing post-disaster weather for {len(coordinates)} coordinates")
            
            # Fetch weather data
            start_time = datetime.now()
            results = self.model.fetch_weather_batch(coordinates, parsed_dates)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Process results and filter variables if requested
            processed_results = []
            successful_count = 0
            
            for i, result in enumerate(results):
                if result is None:
                    processed_results.append({
                        'latitude': coordinates[i]['latitude'],
                        'longitude': coordinates[i]['longitude'], 
                        'disaster_date': parsed_dates[i].strftime('%Y-%m-%d'),
                        'error': 'Failed to fetch weather data',
                        'success': False
                    })
                else:
                    # Filter variables if specified
                    if variables:
                        filtered_result = {
                            'latitude': result['latitude'],
                            'longitude': result['longitude'],
                            'disaster_date': result['disaster_date'],
                            'post_start_date': result['post_start_date'],
                            'post_end_date': result['post_end_date'],
                            'days_fetched': result['days_fetched'],
                            'success': True
                        }
                        
                        for var in variables:
                            if var in result:
                                filtered_result[var] = result[var]
                                # Include statistics if available
                                for suffix in ['_mean', '_std', '_min', '_max', '_missing_days']:
                                    stat_key = var + suffix
                                    if stat_key in result:
                                        filtered_result[stat_key] = result[stat_key]
                        
                        processed_results.append(filtered_result)
                    else:
                        result['success'] = True
                        processed_results.append(result)
                    
                    successful_count += 1
            
            # Update statistics
            self.processing_stats['coordinates_processed'] += len(coordinates)
            self.processing_stats['successful_requests'] += 1 if successful_count > 0 else 0
            if successful_count == 0:
                self.processing_stats['failed_requests'] += 1
            
            return {
                'success': True,
                'data': processed_results,
                'message': f"Successfully processed {successful_count}/{len(coordinates)} coordinates",
                'metadata': {
                    'coordinates_processed': len(coordinates),
                    'successful_extractions': successful_count,
                    'failed_extractions': len(coordinates) - successful_count,
                    'processing_time_seconds': round(processing_time, 3),
                    'variables_extracted': len(variables) if variables else len(self.model.WEATHER_FIELDS),
                    'days_per_coordinate': self.model.days_after_disaster,
                    'extraction_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Post-disaster weather processing error: {e}")
            self.processing_stats['failed_requests'] += 1
            return {
                'success': False,
                'error': f"Processing error: {str(e)}",
                'data': None
            }
    
    def get_available_variables(self) -> Dict[str, Any]:
        """Get available post-disaster weather variables"""
        try:
            variables = self.model.get_available_variables()
            
            return {
                'success': True,
                'variables': variables,
                'total_variables': len(variables),
                'days_per_variable': self.model.days_after_disaster,
                'message': f"{len(variables)} post-disaster weather variables available"
            }
            
        except Exception as e:
            logger.error(f"Error getting available variables: {e}")
            return {
                'success': False,
                'error': f"Failed to get variables: {str(e)}",
                'variables': {}
            }
    
    def export_to_dataframe(self, weather_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export weather data to pandas DataFrame format"""
        try:
            if not weather_data:
                return {
                    'success': False,
                    'error': 'No data to export',
                    'dataframe': None
                }
            
            # Flatten the data for DataFrame creation
            flattened_rows = []
            
            for coord_data in weather_data:
                if not coord_data.get('success', False):
                    continue
                
                base_row = {
                    'latitude': coord_data['latitude'],
                    'longitude': coord_data['longitude'],
                    'disaster_date': coord_data['disaster_date'],
                    'post_start_date': coord_data['post_start_date'],
                    'post_end_date': coord_data['post_end_date'],
                    'days_fetched': coord_data['days_fetched']
                }
                
                # Add weather variables (time series and statistics)
                for key, value in coord_data.items():
                    if key.startswith('POST_'):
                        base_row[key] = value
                
                flattened_rows.append(base_row)
            
            if not flattened_rows:
                return {
                    'success': False,
                    'error': 'No successful weather extractions to export',
                    'dataframe': None
                }
            
            df = pd.DataFrame(flattened_rows)
            
            return {
                'success': True,
                'dataframe': df,
                'shape': df.shape,
                'columns': list(df.columns),
                'message': f"Exported {len(df)} coordinates to DataFrame"
            }
            
        except Exception as e:
            logger.error(f"DataFrame export error: {e}")
            return {
                'success': False,
                'error': f"Export error: {str(e)}",
                'dataframe': None
            }
    
    def export_to_file(self, weather_data: List[Dict[str, Any]], 
                      filepath: str, 
                      file_format: str = 'json') -> Dict[str, Any]:
        """
        Export weather data to file
        
        Args:
            weather_data: Weather extraction results
            filepath: Target file path
            file_format: Export format ('json', 'csv', 'xlsx')
        """
        try:
            if not weather_data:
                return {
                    'success': False,
                    'error': 'No data to export'
                }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if file_format.lower() == 'json':
                # Handle NaN values for JSON serialization
                def convert_nan(obj):
                    if isinstance(obj, float) and pd.isna(obj):
                        return None
                    elif isinstance(obj, list):
                        return [convert_nan(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {key: convert_nan(value) for key, value in obj.items()}
                    return obj
                
                cleaned_data = convert_nan(weather_data)
                
                with open(filepath, 'w') as f:
                    json.dump(cleaned_data, f, indent=2, default=str)
                
            elif file_format.lower() in ['csv', 'xlsx']:
                df_result = self.export_to_dataframe(weather_data)
                if not df_result['success']:
                    return df_result
                
                df = df_result['dataframe']
                
                if file_format.lower() == 'csv':
                    df.to_csv(filepath, index=False)
                else:  # xlsx
                    df.to_excel(filepath, index=False, engine='openpyxl')
            
            else:
                return {
                    'success': False,
                    'error': f"Unsupported file format: {file_format}. Supported: json, csv, xlsx"
                }
            
            file_size = os.path.getsize(filepath)
            
            return {
                'success': True,
                'filepath': filepath,
                'file_format': file_format,
                'file_size_bytes': file_size,
                'coordinates_exported': len([d for d in weather_data if d.get('success', False)]),
                'message': f"Successfully exported to {filepath}"
            }
            
        except Exception as e:
            logger.error(f"File export error: {e}")
            return {
                'success': False,
                'error': f"Export error: {str(e)}"
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get service processing statistics"""
        try:
            model_stats = self.model.get_processing_stats()
            
            total_requests = self.processing_stats['total_requests']
            success_rate = 0
            if total_requests > 0:
                success_rate = (self.processing_stats['successful_requests'] / total_requests) * 100
            
            # Determine service health
            if total_requests == 0:
                service_status = 'ready'
            elif success_rate >= 80:
                service_status = 'healthy'
            elif success_rate >= 50:
                service_status = 'degraded'
            else:
                service_status = 'unhealthy'
            
            return {
                'success': True,
                'statistics': {
                    **self.processing_stats,
                    'success_rate': round(success_rate, 2),
                    'service_status': service_status,
                    'model_stats': model_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {
                'success': False,
                'error': f"Failed to get statistics: {str(e)}",
                'statistics': {}
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get overall service health status"""
        try:
            stats_result = self.get_processing_statistics()
            if not stats_result['success']:
                return {
                    'service_name': 'post_disaster_weather',
                    'status': 'error',
                    'message': 'Failed to get statistics'
                }
            
            stats = stats_result['statistics']
            status = stats['service_status']
            
            return {
                'service_name': 'post_disaster_weather',
                'status': 'healthy' if status in ['ready', 'healthy'] else status,
                'initialized': True,
                'total_requests': stats['total_requests'],
                'success_rate': stats['success_rate'],
                'coordinates_processed': stats['coordinates_processed'],
                'last_request': stats['last_request_time'],
                'message': f"Service is {status}"
            }
            
        except Exception as e:
            logger.error(f"Service status error: {e}")
            return {
                'service_name': 'post_disaster_weather',
                'status': 'error',
                'initialized': False,
                'message': f"Service error: {str(e)}"
            }
    
    def validate_batch_request(self, request_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate batch processing request"""
        try:
            if 'coordinates' not in request_data:
                return False, "'coordinates' field is required"
            
            if 'disaster_dates' not in request_data:
                return False, "'disaster_dates' field is required"
            
            coordinates = request_data['coordinates']
            disaster_dates = request_data['disaster_dates']
            
            if not isinstance(coordinates, list) or not coordinates:
                return False, "'coordinates' must be a non-empty list"
            
            if not isinstance(disaster_dates, list) or not disaster_dates:
                return False, "'disaster_dates' must be a non-empty list"
            
            # Check length compatibility
            if len(disaster_dates) != 1 and len(disaster_dates) != len(coordinates):
                return False, f"Must provide either 1 disaster date or {len(coordinates)} dates for {len(coordinates)} coordinates"
            
            # Validate batch size
            if len(coordinates) > 1000:
                return False, f"Batch size limit exceeded: {len(coordinates)} coordinates (max: 1000)"
            
            return True, "Batch request is valid"
            
        except Exception as e:
            logger.error(f"Batch validation error: {e}")
            return False, f"Validation error: {str(e)}"