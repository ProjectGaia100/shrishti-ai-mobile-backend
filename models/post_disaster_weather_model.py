"""
Post-Disaster Weather Data Model for HazardGuard System
Fetches weather data for 60 days AFTER disaster occurrence using NASA POWER API
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

logger = logging.getLogger(__name__)

class PostDisasterWeatherModel:
    """Model for fetching post-disaster weather data from NASA POWER API"""
    
    # Post-disaster weather variables (17 total) with POST_ prefix
    WEATHER_FIELDS = {
        # Original 6 core variables
        'T2M': 'POST_temperature_C',
        'RH2M': 'POST_humidity_%', 
        'WS2M': 'POST_wind_speed_mps',
        'PRECTOTCORR': 'POST_precipitation_mm',
        'PS': 'POST_surface_pressure_hPa',
        'ALLSKY_SFC_SW_DWN': 'POST_solar_radiation_wm2',
        # Additional 11 variables for comprehensive analysis
        'T2M_MAX': 'POST_temperature_max_C',
        'T2M_MIN': 'POST_temperature_min_C', 
        'QV2M': 'POST_specific_humidity_g_kg',
        'T2MDEW': 'POST_dew_point_C',
        'WS10M': 'POST_wind_speed_10m_mps',
        'CLOUD_AMT': 'POST_cloud_amount_%',
        'SLP': 'POST_sea_level_pressure_hPa',
        'GWETTOP': 'POST_surface_soil_wetness_%',
        'WD10M': 'POST_wind_direction_10m_degrees',
        'EVPTRNS': 'POST_evapotranspiration_wm2',
        'GWETROOT': 'POST_root_zone_soil_moisture_%'
    }
    
    # NASA POWER API fill values that should be treated as NaN
    NASA_FILL_VALUES = [-999, -999.0, -99999, -99999.0]
    
    def __init__(self, 
                 days_after_disaster: int = 60,
                 max_workers: int = 1, 
                 retry_limit: int = 5,
                 retry_delay: int = 15,
                 rate_limit_pause: int = 900,
                 request_delay: float = 0.5):
        """
        Initialize post-disaster weather model
        
        Args:
            days_after_disaster: Number of days to fetch after disaster (default: 60)
            max_workers: Maximum concurrent API requests (default: 1)
            retry_limit: Maximum retry attempts per request (default: 5) 
            retry_delay: Delay between retries in seconds (default: 15)
            rate_limit_pause: Pause duration for rate limits in seconds (default: 900)
            request_delay: Delay between requests in seconds (default: 0.5)
        """
        self.days_after_disaster = days_after_disaster
        self.max_workers = max_workers
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self.rate_limit_pause = rate_limit_pause
        self.request_delay = request_delay
        
        self.api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        logger.info(f"Initialized PostDisasterWeatherModel: {days_after_disaster} days, {len(self.WEATHER_FIELDS)} variables")
    
    def validate_coordinates(self, coordinates: List[Dict[str, float]]) -> Tuple[bool, str]:
        """Validate coordinate format and ranges"""
        try:
            if not coordinates or not isinstance(coordinates, list):
                return False, "Coordinates must be a non-empty list"
            
            for i, coord in enumerate(coordinates):
                if not isinstance(coord, dict):
                    return False, f"Coordinate {i} must be a dictionary"
                
                if 'latitude' not in coord or 'longitude' not in coord:
                    return False, f"Coordinate {i} must have 'latitude' and 'longitude' keys"
                
                try:
                    lat = float(coord['latitude'])
                    lon = float(coord['longitude'])
                    
                    if not (-90 <= lat <= 90):
                        return False, f"Coordinate {i} latitude {lat} out of range (-90 to 90)"
                    if not (-180 <= lon <= 180):
                        return False, f"Coordinate {i} longitude {lon} out of range (-180 to 180)"
                        
                except (TypeError, ValueError):
                    return False, f"Coordinate {i} has invalid latitude/longitude values"
            
            return True, "Coordinates are valid"
            
        except Exception as e:
            logger.error(f"Coordinate validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_disaster_date(self, disaster_date: Union[str, datetime]) -> Tuple[bool, str, Optional[datetime]]:
        """Validate and parse disaster date"""
        try:
            if isinstance(disaster_date, str):
                # Try multiple date formats
                date_formats = [
                    '%Y-%m-%d',
                    '%Y/%m/%d', 
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y-%m-%d %H:%M:%S',
                    '%m-%d-%Y',
                    '%d-%m-%Y'
                ]
                
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(disaster_date, fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    return False, f"Unable to parse date '{disaster_date}'", None
                        
            elif isinstance(disaster_date, datetime):
                parsed_date = disaster_date
            else:
                return False, "Disaster date must be string or datetime", None
            
            # Check if end date would be too recent (API has ~7 day lag)
            end_date = parsed_date + timedelta(days=self.days_after_disaster)
            current_date = datetime.now() - timedelta(days=7)
            
            if end_date > current_date:
                return False, f"End date {end_date.date()} is too recent (API has ~7 day lag)", None
            
            return True, "Date is valid", parsed_date
            
        except Exception as e:
            logger.error(f"Date validation error: {e}")
            return False, f"Date validation error: {str(e)}", None
    
    def clean_nasa_values(self, values: List[float]) -> List[Optional[float]]:
        """Clean NASA API values, converting fill values to NaN"""
        if not values:
            return [np.nan] * self.days_after_disaster
        
        cleaned = []
        for value in values:
            if value in self.NASA_FILL_VALUES or pd.isna(value):
                cleaned.append(np.nan)
            else:
                # Convert numpy types to native Python types for JSON serialization
                if hasattr(value, 'item'):
                    cleaned.append(float(value.item()))
                else:
                    cleaned.append(float(value))
        
        # Ensure we have exactly the right number of days
        if len(cleaned) < self.days_after_disaster:
            cleaned.extend([np.nan] * (self.days_after_disaster - len(cleaned)))
        elif len(cleaned) > self.days_after_disaster:
            cleaned = cleaned[:self.days_after_disaster]
        
        return cleaned
    
    def fetch_weather_for_coordinate(self, coordinate: Dict[str, float], disaster_date: datetime) -> Dict[str, Any]:
        """Fetch post-disaster weather data for a single coordinate"""
        try:
            lat = float(coordinate['latitude'])
            lon = float(coordinate['longitude'])
            
            # Calculate post-disaster date range
            post_start_date = disaster_date + timedelta(days=1)  # Start day after disaster
            post_end_date = post_start_date + timedelta(days=self.days_after_disaster - 1)
            
            logger.debug(f"Fetching post-disaster weather for lat={lat}, lon={lon}, dates={post_start_date.date()} to {post_end_date.date()}")
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start": post_start_date.strftime("%Y%m%d"),
                "end": post_end_date.strftime("%Y%m%d"),
                "community": "RE",
                "format": "JSON",
                "parameters": ','.join(self.WEATHER_FIELDS.keys())
            }
            
            for attempt in range(self.retry_limit):
                try:
                    # Add delay to avoid overwhelming API
                    time.sleep(self.request_delay)
                    
                    response = requests.get(self.api_url, params=params, timeout=60)
                    self.request_count += 1
                    
                    if response.status_code == 429:
                        logger.warning(f"Rate limit hit (429). Pausing {self.rate_limit_pause}s...")
                        time.sleep(self.rate_limit_pause)
                        continue
                    
                    response.raise_for_status()
                    data = response.json().get("properties", {}).get("parameter", {})
                    
                    if not data:
                        logger.warning(f"No data returned from API for lat={lat}, lon={lon}")
                        self.failure_count += 1
                        return None
                    
                    # Process weather data
                    result = {
                        'latitude': lat,
                        'longitude': lon,
                        'disaster_date': disaster_date.strftime('%Y-%m-%d'),
                        'post_start_date': post_start_date.strftime('%Y-%m-%d'),
                        'post_end_date': post_end_date.strftime('%Y-%m-%d'),
                        'days_fetched': self.days_after_disaster
                    }
                    
                    for nasa_key, col_name in self.WEATHER_FIELDS.items():
                        raw_values = list(data.get(nasa_key, {}).values())
                        cleaned_values = self.clean_nasa_values(raw_values)
                        result[col_name] = cleaned_values
                        
                        # Add summary statistics
                        valid_values = [v for v in cleaned_values if not pd.isna(v)]
                        if valid_values:
                            result[f"{col_name}_mean"] = float(np.mean(valid_values))
                            result[f"{col_name}_std"] = float(np.std(valid_values))
                            result[f"{col_name}_min"] = float(np.min(valid_values))
                            result[f"{col_name}_max"] = float(np.max(valid_values))
                            result[f"{col_name}_missing_days"] = int(self.days_after_disaster - len(valid_values))
                        else:
                            result[f"{col_name}_mean"] = np.nan
                            result[f"{col_name}_std"] = np.nan
                            result[f"{col_name}_min"] = np.nan
                            result[f"{col_name}_max"] = np.nan
                            result[f"{col_name}_missing_days"] = int(self.days_after_disaster)
                    
                    self.success_count += 1
                    logger.debug(f"Successfully fetched post-disaster weather for lat={lat}, lon={lon}")
                    return result
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request error (attempt {attempt + 1}/{self.retry_limit}): {e}")
                    if attempt < self.retry_limit - 1:
                        time.sleep(self.retry_delay)
                    else:
                        self.failure_count += 1
                        logger.error(f"Failed to fetch after {self.retry_limit} attempts for lat={lat}, lon={lon}")
                        return None
                        
        except Exception as e:
            logger.error(f"Critical error fetching weather for coordinate {coordinate}: {e}")
            self.failure_count += 1
            return None
    
    def fetch_weather_batch(self, coordinates: List[Dict[str, float]], disaster_dates: List[datetime]) -> List[Optional[Dict[str, Any]]]:
        """Fetch post-disaster weather data for multiple coordinates"""
        try:
            if len(coordinates) != len(disaster_dates):
                raise ValueError("Number of coordinates must match number of disaster dates")
            
            logger.info(f"Fetching post-disaster weather for {len(coordinates)} coordinates using {self.max_workers} workers")
            
            results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(self.fetch_weather_for_coordinate, coord, date): i 
                    for i, (coord, date) in enumerate(zip(coordinates, disaster_dates))
                }
                
                # Collect results in original order
                indexed_results = {}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        indexed_results[index] = result
                    except Exception as e:
                        logger.error(f"Error processing coordinate {index}: {e}")
                        indexed_results[index] = None
                
                # Sort by index to maintain order
                results = [indexed_results[i] for i in range(len(coordinates))]
            
            logger.info(f"Completed batch processing: {self.success_count} successes, {self.failure_count} failures")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [None] * len(coordinates)
    
    def get_available_variables(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available post-disaster weather variables"""
        variable_info = {}
        
        for nasa_key, col_name in self.WEATHER_FIELDS.items():
            variable_info[col_name] = {
                'nasa_parameter': nasa_key,
                'description': self._get_variable_description(nasa_key),
                'unit': self._get_variable_unit(col_name),
                'type': 'time_series',
                'days': self.days_after_disaster,
                'includes_statistics': True
            }
        
        return variable_info
    
    def _get_variable_description(self, nasa_key: str) -> str:
        """Get human-readable description for NASA parameter"""
        descriptions = {
            'T2M': 'Temperature at 2 Meters',
            'RH2M': 'Relative Humidity at 2 Meters',
            'WS2M': 'Wind Speed at 2 Meters',
            'PRECTOTCORR': 'Precipitation Corrected',
            'PS': 'Surface Pressure',
            'ALLSKY_SFC_SW_DWN': 'All Sky Surface Shortwave Downward Irradiance',
            'T2M_MAX': 'Temperature at 2 Meters Maximum',
            'T2M_MIN': 'Temperature at 2 Meters Minimum',
            'QV2M': 'Specific Humidity at 2 Meters',
            'T2MDEW': 'Dew/Frost Point at 2 Meters',
            'WS10M': 'Wind Speed at 10 Meters',
            'CLOUD_AMT': 'Cloud Amount',
            'SLP': 'Sea Level Pressure',
            'GWETTOP': 'Surface Soil Wetness',
            'WD10M': 'Wind Direction at 10 Meters',
            'EVPTRNS': 'Evapotranspiration Energy Flux',
            'GWETROOT': 'Root Zone Soil Wetness'
        }
        return descriptions.get(nasa_key, nasa_key)
    
    def _get_variable_unit(self, col_name: str) -> str:
        """Get unit for variable from column name"""
        if 'temperature' in col_name.lower():
            return '°C'
        elif 'humidity' in col_name.lower():
            return '%'
        elif 'wind_speed' in col_name.lower():
            return 'm/s'
        elif 'precipitation' in col_name.lower():
            return 'mm'
        elif 'pressure' in col_name.lower():
            return 'hPa'
        elif 'radiation' in col_name.lower() or 'evapotranspiration' in col_name.lower():
            return 'W/m²'
        elif 'cloud' in col_name.lower() or 'wetness' in col_name.lower() or 'moisture' in col_name.lower():
            return '%'
        elif 'dew_point' in col_name.lower():
            return '°C'
        elif 'wind_direction' in col_name.lower():
            return 'degrees'
        elif 'humidity_g_kg' in col_name:
            return 'g/kg'
        else:
            return 'units'
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_requests = self.request_count
        success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate': round(success_rate, 2),
            'days_per_request': self.days_after_disaster,
            'total_variables': len(self.WEATHER_FIELDS),
            'api_endpoint': self.api_url.split('/')[-1]
        }