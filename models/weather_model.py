"""
Weather Data Model
Defines weather data structure and validation
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

@dataclass
class WeatherDataPoint:
    """Single day weather data point"""
    date: str
    temperature_C: Optional[float] = None
    humidity_perc: Optional[float] = None
    wind_speed_mps: Optional[float] = None
    precipitation_mm: Optional[float] = None
    surface_pressure_hPa: Optional[float] = None
    solar_radiation_wm2: Optional[float] = None
    temperature_max_C: Optional[float] = None
    temperature_min_C: Optional[float] = None
    specific_humidity_g_kg: Optional[float] = None
    dew_point_C: Optional[float] = None
    wind_speed_10m_mps: Optional[float] = None
    cloud_amount_perc: Optional[float] = None
    sea_level_pressure_hPa: Optional[float] = None
    surface_soil_wetness_perc: Optional[float] = None
    wind_direction_10m_degrees: Optional[float] = None
    evapotranspiration_wm2: Optional[float] = None
    root_zone_soil_moisture_perc: Optional[float] = None

@dataclass
class WeatherRequest:
    """Weather data request parameters"""
    latitude: float
    longitude: float
    disaster_date: str  # YYYY-MM-DD format
    days_before: int = 60
    
    def validate(self) -> Dict[str, Any]:
        """Validate request parameters"""
        errors = []
        
        # Validate coordinates
        if not (-90 <= self.latitude <= 90):
            errors.append(f"Latitude {self.latitude} out of range (-90 to 90)")
        if not (-180 <= self.longitude <= 180):
            errors.append(f"Longitude {self.longitude} out of range (-180 to 180)")
            
        # Validate date
        try:
            disaster_datetime = datetime.strptime(self.disaster_date, '%Y-%m-%d')
            # Check if date is not too recent (NASA has ~7 day lag)
            current_date = datetime.now() - timedelta(days=7)
            if disaster_datetime > current_date:
                errors.append(f"Disaster date {self.disaster_date} too recent (NASA has ~7 day lag)")
        except ValueError:
            errors.append(f"Invalid date format. Use YYYY-MM-DD")
            
        # Validate days_before
        if not (1 <= self.days_before <= 365):
            errors.append(f"days_before must be between 1 and 365, got {self.days_before}")
            
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

class WeatherDataModel:
    """Weather data model with validation and processing"""
    
    # NASA POWER weather fields mapping
    WEATHER_FIELDS = {
        # Original 6 variables
        'T2M': 'temperature_C',
        'RH2M': 'humidity_perc',
        'WS2M': 'wind_speed_mps', 
        'PRECTOTCORR': 'precipitation_mm',
        'PS': 'surface_pressure_hPa',
        'ALLSKY_SFC_SW_DWN': 'solar_radiation_wm2',
        # Additional 11 variables for better disaster prediction
        'T2M_MAX': 'temperature_max_C',
        'T2M_MIN': 'temperature_min_C',
        'QV2M': 'specific_humidity_g_kg',
        'T2MDEW': 'dew_point_C',
        'WS10M': 'wind_speed_10m_mps',
        'CLOUD_AMT': 'cloud_amount_perc',
        'SLP': 'sea_level_pressure_hPa',
        'GWETTOP': 'surface_soil_wetness_perc',
        'WD10M': 'wind_direction_10m_degrees',
        'EVPTRNS': 'evapotranspiration_wm2',
        'GWETROOT': 'root_zone_soil_moisture_perc'
    }
    
    # NASA POWER fill values that should be converted to NaN
    FILL_VALUES = [-999, -999.0, -99999, -99999.0]
    
    @classmethod
    def process_raw_data(cls, raw_data: Dict[str, Dict[str, float]], 
                        days_count: int) -> Dict[str, List[Optional[float]]]:
        """
        Process raw NASA POWER API response into structured format
        
        Args:
            raw_data: Raw response from NASA POWER API
            days_count: Expected number of days
            
        Returns:
            Dictionary with processed weather data lists (chronologically ordered)
        """
        processed = {}
        
        for nasa_key, col_name in cls.WEATHER_FIELDS.items():
            raw_values = raw_data.get(nasa_key, {})
            
            # FIXED: Sort date keys chronologically to ensure proper order
            date_keys = sorted(raw_values.keys()) if raw_values else []
            
            # Convert to list of values, handling fill values
            values = []
            for i in range(days_count):
                if i < len(date_keys):
                    date_key = date_keys[i]
                    value = raw_values[date_key]
                    # Convert fill values to NaN
                    if value in cls.FILL_VALUES:
                        values.append(None)
                    else:
                        values.append(float(value) if value is not None else None)
                else:
                    values.append(None)  # Missing data as NaN
            
            processed[col_name] = values
            
        return processed
    
    @classmethod
    def create_time_series_dataframe(cls, weather_data: Dict[str, List], 
                                   disaster_date: str, days_before: int) -> pd.DataFrame:
        """
        Create time series DataFrame from weather data
        
        Args:
            weather_data: Processed weather data dictionary
            disaster_date: Disaster date string (YYYY-MM-DD)
            days_before: Number of days before disaster (includes disaster date)
            
        Returns:
            DataFrame with time series weather data
        """
        disaster_dt = datetime.strptime(disaster_date, '%Y-%m-%d')
        # FIXED: End date is disaster date, start is (days_before-1) days before
        end_date = disaster_dt
        start_date = end_date - timedelta(days=days_before - 1)
        
        # Generate date range: from start_date to end_date (inclusive)
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        date_strings = [dt.strftime('%Y-%m-%d') for dt in date_range]
        
        # Create DataFrame
        df = pd.DataFrame({'date': date_strings})
        
        # Add weather data columns
        for col_name, values in weather_data.items():
            # Ensure we have exactly the right number of values
            padded_values = values[:len(date_range)] + [None] * max(0, len(date_range) - len(values))
            df[col_name] = padded_values[:len(date_range)]
            
        return df
    
    @classmethod
    def validate_weather_data(cls, weather_data: Dict[str, List], 
                            expected_days: int) -> Dict[str, Any]:
        """
        Validate processed weather data
        
        Args:
            weather_data: Processed weather data
            expected_days: Expected number of days
            
        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []
        
        # Check if all required fields are present
        missing_fields = set(cls.WEATHER_FIELDS.values()) - set(weather_data.keys())
        if missing_fields:
            errors.append(f"Missing weather fields: {missing_fields}")
            
        # Check data completeness
        for field, values in weather_data.items():
            if len(values) != expected_days:
                warnings.append(f"{field}: expected {expected_days} values, got {len(values)}")
                
            # Check for data availability
            non_null_count = sum(1 for v in values if v is not None)
            completeness = (non_null_count / len(values)) * 100 if values else 0
            
            if completeness < 50:
                warnings.append(f"{field}: low data completeness ({completeness:.1f}%)")
                
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'data_quality': {
                'total_fields': len(weather_data),
                'expected_days': expected_days,
                'completeness_summary': {
                    field: sum(1 for v in values if v is not None) / len(values) * 100 
                    if values else 0
                    for field, values in weather_data.items()
                }
            }
        }