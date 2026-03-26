"""
NASA POWER Weather Service
Handles weather data fetching from NASA POWER API
"""
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from models.weather_model import WeatherRequest, WeatherDataModel

class NASAPowerService:
    """Service for fetching weather data from NASA POWER API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # API settings
        self.max_retries = 5
        self.retry_delay = 15  # seconds
        self.rate_limit_pause = 900  # 15 minutes for rate limit (429)
        self.request_delay = 0.5  # delay between requests
        self.timeout = 60  # request timeout
        
        self.initialized = True
        
    def fetch_weather_data(self, request: WeatherRequest) -> Tuple[bool, Dict[str, Any]]:
        """
        Fetch weather data from NASA POWER API
        
        Args:
            request: Weather data request with coordinates and date
            
        Returns:
            Tuple of (success: bool, result: dict)
        """
        try:
            # Validate request
            validation = request.validate()
            if not validation['valid']:
                return False, {
                    'error': 'Invalid request parameters',
                    'details': validation['errors'],
                    'status': 'validation_error'
                }
            
            # Calculate date range
            disaster_date = datetime.strptime(request.disaster_date, '%Y-%m-%d')
            end_date = disaster_date
            start_date = end_date - timedelta(days=request.days_before - 1)
            
            # Prepare API parameters
            params = {
                "latitude": request.latitude,
                "longitude": request.longitude, 
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "community": "RE",
                "format": "JSON",
                "parameters": ','.join(WeatherDataModel.WEATHER_FIELDS.keys())
            }
            
            self.logger.info(f"Fetching weather data for lat={request.latitude}, lon={request.longitude}, "
                           f"date_range={start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Make API request with retries
            for attempt in range(self.max_retries):
                try:
                    # Rate limiting
                    time.sleep(self.request_delay)
                    
                    self.logger.debug(f"NASA POWER API request (attempt {attempt + 1}): {params}")
                    response = requests.get(self.base_url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    
                    # Parse response
                    json_data = response.json()
                    raw_weather_data = json_data.get("properties", {}).get("parameter", {})
                    
                    if not raw_weather_data:
                        self.logger.warning(f"Empty weather data response for coordinates ({request.latitude}, {request.longitude})")
                        return False, {
                            'error': 'No weather data returned from NASA POWER API',
                            'status': 'no_data'
                        }
                    
                    # Process raw data
                    processed_data = WeatherDataModel.process_raw_data(raw_weather_data, request.days_before)
                    
                    # Validate processed data
                    validation_result = WeatherDataModel.validate_weather_data(processed_data, request.days_before)
                    
                    self.logger.info(f"Successfully fetched weather data: {len(processed_data)} fields, "
                                   f"{request.days_before} days")
                    
                    return True, {
                        'weather_data': processed_data,
                        'validation': validation_result,
                        'metadata': {
                            'latitude': request.latitude,
                            'longitude': request.longitude,
                            'disaster_date': request.disaster_date,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d'),
                            'days_count': request.days_before,
                            'api_response_time': response.elapsed.total_seconds(),
                            'data_quality': validation_result.get('data_quality', {})
                        },
                        'status': 'success'
                    }
                    
                except requests.exceptions.HTTPError as e:
                    if e.response and e.response.status_code == 429:
                        self.logger.warning(f"Rate limit hit (429). Pausing {self.rate_limit_pause}s...")
                        time.sleep(self.rate_limit_pause)
                        continue  # Don't count as failed attempt
                    else:
                        error_msg = f"HTTP Error {e.response.status_code}: {e.response.text if e.response else 'No response'}"
                        self.logger.error(error_msg)
                        
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            return False, {
                                'error': error_msg,
                                'status': 'api_error',
                                'attempts': attempt + 1
                            }
                            
                except requests.exceptions.Timeout:
                    error_msg = f"Request timeout after {self.timeout}s"
                    self.logger.error(error_msg)
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return False, {
                            'error': error_msg,
                            'status': 'timeout',
                            'attempts': attempt + 1
                        }
                        
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    self.logger.error(error_msg)
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return False, {
                            'error': error_msg,
                            'status': 'unknown_error',
                            'attempts': attempt + 1
                        }
            
            # If we get here, all retries failed
            return False, {
                'error': f'All {self.max_retries} attempts failed',
                'status': 'max_retries_exceeded'
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in fetch_weather_data: {str(e)}")
            return False, {
                'error': f'Critical service error: {str(e)}',
                'status': 'service_error'
            }
    
    def get_weather_for_coordinates(self, latitude: float, longitude: float, 
                                  disaster_date: str, days_before: int = 60) -> Dict[str, Any]:
        """
        Convenience method to get weather data for coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate 
            disaster_date: Disaster date in YYYY-MM-DD format
            days_before: Number of days before disaster to fetch
            
        Returns:
            Weather data result dictionary
        """
        request = WeatherRequest(
            latitude=latitude,
            longitude=longitude,
            disaster_date=disaster_date,
            days_before=days_before
        )
        
        success, result = self.fetch_weather_data(request)
        return {
            'success': success,
            **result
        }
    
    def batch_fetch_weather_data(self, requests_list: list) -> Dict[str, Any]:
        """
        Fetch weather data for multiple locations (with rate limiting)
        
        Args:
            requests_list: List of WeatherRequest objects
            
        Returns:
            Batch processing results
        """
        results = []
        successful = 0
        failed = 0
        
        self.logger.info(f"Starting batch fetch for {len(requests_list)} locations")
        
        for i, request in enumerate(requests_list):
            self.logger.info(f"Processing batch item {i + 1}/{len(requests_list)}")
            
            success, result = self.fetch_weather_data(request)
            
            results.append({
                'index': i,
                'request': {
                    'latitude': request.latitude,
                    'longitude': request.longitude,
                    'disaster_date': request.disaster_date,
                    'days_before': request.days_before
                },
                'success': success,
                'result': result
            })
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Rate limiting between requests
            if i < len(requests_list) - 1:  # Don't sleep after last request
                time.sleep(self.request_delay)
        
        return {
            'batch_summary': {
                'total_requests': len(requests_list),
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / len(requests_list)) * 100 if requests_list else 0
            },
            'results': results,
            'status': 'completed'
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service health and configuration"""
        return {
            'service': 'NASA POWER Weather Service',
            'initialized': self.initialized,
            'base_url': self.base_url,
            'configuration': {
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay,
                'rate_limit_pause': self.rate_limit_pause,
                'request_delay': self.request_delay,
                'timeout': self.timeout
            },
            'supported_fields': list(WeatherDataModel.WEATHER_FIELDS.values()),
            'field_count': len(WeatherDataModel.WEATHER_FIELDS)
        }