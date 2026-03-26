"""
Feature Engineering Model
Handles weather feature engineering computations with proper NaN handling

v3 FIX: Uses training-dataset global statistics for feature normalization
       instead of per-window stats (which inflated disaster features for
       normal weather, causing the model to always predict 'Disaster').
"""
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

class WeatherFeatureModel:
    """Model for weather feature engineering operations"""
    
    # Path to training-dataset global statistics
    # These stats were computed across the ENTIRE training dataset (~123k rows × 60 days)
    # and MUST be used for feature engineering to match the training distribution.
    # Without them, per-window stats inflate disaster-related features for normal weather.
    TRAINING_STATS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'hazardguard', 'shared', 'training_global_stats.json'
    )
    
    # Cached training stats (loaded once)
    _training_stats = None
    
    # Original 17 weather fields from NASA POWER
    WEATHER_FIELDS = [
        'temperature_C',
        'humidity_perc',
        'wind_speed_mps', 
        'precipitation_mm',
        'surface_pressure_hPa',
        'solar_radiation_wm2',
        'temperature_max_C',
        'temperature_min_C',
        'specific_humidity_g_kg',
        'dew_point_C',
        'wind_speed_10m_mps',
        'cloud_amount_perc',
        'sea_level_pressure_hPa',
        'surface_soil_wetness_perc',
        'wind_direction_10m_degrees',
        'evapotranspiration_wm2',
        'root_zone_soil_moisture_perc'
    ]
    
    # 19 engineered features (excluding precip_intensity_mm_day as requested)
    ENGINEERED_FEATURES = [
        'temp_normalized',
        'temp_range',
        'discomfort_index',
        'heat_index',
        'wind_precip_interaction',
        'solar_temp_ratio',
        'pressure_anomaly',
        'high_precip_flag',
        'adjusted_humidity',
        'wind_chill',
        'solar_radiation_anomaly',
        'weather_severity_score', 
        'moisture_stress_index',
        'evaporation_deficit',
        'soil_saturation_index',
        'atmospheric_instability',
        'drought_indicator',
        'flood_risk_score',
        'storm_intensity_index'
    ]
    
    # Feature descriptions for documentation
    FEATURE_DESCRIPTIONS = {
        'temp_normalized': 'Temperature normalized between min/max',
        'temp_range': 'Diurnal temperature range (max - min)',
        'discomfort_index': 'Temperature-Humidity Index (THI)',
        'heat_index': 'Apparent temperature accounting for humidity',
        'wind_precip_interaction': 'Wind speed × precipitation interaction',
        'solar_temp_ratio': 'Solar radiation to temperature ratio',
        'pressure_anomaly': 'Deviation from mean surface pressure',
        'high_precip_flag': 'Binary flag for precipitation >50mm',
        'adjusted_humidity': 'Humidity adjusted for temperature',
        'wind_chill': 'Perceived temperature with wind effect',
        'solar_radiation_anomaly': 'Deviation from mean solar radiation',
        'weather_severity_score': 'Composite severity index (0-1)',
        'moisture_stress_index': 'Evaporation vs precipitation stress',
        'evaporation_deficit': 'Deviation from mean evapotranspiration',
        'soil_saturation_index': 'Combined surface + root zone moisture',
        'atmospheric_instability': 'Pressure difference + temperature range',
        'drought_indicator': 'Low precip + high temp + low soil moisture',
        'flood_risk_score': 'High precip + saturated soil + low evap',
        'storm_intensity_index': 'Wind + precipitation + pressure drop'
    }
    
    @classmethod
    def validate_weather_data(cls, weather_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Validate weather data for feature engineering
        
        Args:
            weather_data: Dictionary with weather field lists
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check required fields
        missing_fields = set(cls.WEATHER_FIELDS) - set(weather_data.keys())
        if missing_fields:
            errors.append(f"Missing required weather fields: {missing_fields}")
        
        # Check list lengths and data quality
        days_count = None
        for field, values in weather_data.items():
            if not isinstance(values, list):
                errors.append(f"Field {field} must be a list, got {type(values)}")
                continue
                
            if days_count is None:
                days_count = len(values)
            elif len(values) != days_count:
                warnings.append(f"Field {field} has {len(values)} values, expected {days_count}")
        
        # Check for all NaN lists
        for field, values in weather_data.items():
            if isinstance(values, list):
                valid_count = sum(1 for v in values if v is not None and not np.isnan(float(v)) if v != -999)
                if valid_count == 0:
                    warnings.append(f"Field {field} contains only NaN/missing values")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'days_count': days_count,
            'field_count': len(weather_data)
        }
    
    @classmethod
    def load_training_stats(cls) -> Dict[str, float]:
        """
        Load global statistics from the training dataset.
        
        These stats were pre-computed across the ENTIRE training dataset
        (~123k rows × 60 days = ~7.4M data points per weather variable).
        Using per-window stats instead leads to feature scale mismatch:
        e.g., precip_max=994mm (training) vs precip_max=10mm (single window)
        which inflates disaster features 100x and causes false positives.
        
        Returns:
            Dictionary of training-dataset global statistics
        """
        logger = logging.getLogger(__name__)
        
        # Return cached stats if already loaded
        if cls._training_stats is not None:
            return cls._training_stats
        
        # Try loading from JSON file
        stats_path = cls.TRAINING_STATS_PATH
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    loaded = json.load(f)
                cls._training_stats = {
                    'temp_min': loaded.get('temp_min', -53.76),
                    'temp_max': loaded.get('temp_max', 44.18),
                    'temp_mean': loaded.get('temp_mean', 21.5325),
                    'temp_max_mean': loaded.get('temp_max_mean', 25.7065),
                    'pressure_mean': loaded.get('pressure_mean', 93.7966),
                    'sea_pressure_mean': loaded.get('sea_pressure_mean', 101.0376),
                    'solar_mean': loaded.get('solar_mean', 4.9125),
                    'precip_max': max(loaded.get('precip_max', 994.16), 1),
                    'wind_max': max(loaded.get('wind_max', 25.32), 1),
                    'evap_mean': loaded.get('evap_mean', 0.5756),
                }
                logger.info(f"[STATS] Loaded training global stats from {stats_path}")
                logger.info(f"   precip_max={cls._training_stats['precip_max']}, "
                           f"wind_max={cls._training_stats['wind_max']}, "
                           f"temp_range=[{cls._training_stats['temp_min']}, {cls._training_stats['temp_max']}]")
                return cls._training_stats
            except Exception as e:
                logger.error(f"[STATS] Error loading training stats: {e}")
        else:
            logger.warning(f"[STATS] Training stats file not found: {stats_path}")
        
        # Fallback: hardcoded values from actual training dataset computation
        # (computed from VALIDATED_LAT-LONG_CLEAN_DEDUPLICATED.xlsx, 123143 rows)
        logger.warning("[STATS] Using hardcoded fallback training stats")
        cls._training_stats = {
            'temp_min': -53.76,
            'temp_max': 44.18,
            'temp_mean': 21.5325,
            'temp_max_mean': 25.7065,
            'pressure_mean': 93.7966,
            'sea_pressure_mean': 101.0376,
            'solar_mean': 4.9125,
            'precip_max': 994.16,
            'wind_max': 25.32,
            'evap_mean': 0.5756,
        }
        return cls._training_stats
    
    @classmethod
    def compute_global_stats(cls, weather_data: Dict[str, List]) -> Dict[str, float]:
        """
        Return global statistics for feature engineering.
        
        FIXED (v3): Now returns training-dataset stats instead of computing
        per-window stats. The per-window approach inflated disaster features
        for normal weather (e.g., precip/precip_max_window ≈ 1.0 vs
        precip/precip_max_training ≈ 0.01) causing 100% disaster predictions.
        
        Args:
            weather_data: Weather data dictionary (ignored — uses training stats)
            
        Returns:
            Dictionary of global statistics from training dataset
        """
        return cls.load_training_stats()
    
    @classmethod
    def safe_float(cls, value, default=np.nan):
        """Safely convert value to float, return default for NaN/invalid values"""
        if value is None or value == -999:
            return default
        try:
            float_val = float(value)
            return float_val if not np.isnan(float_val) else default
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def compute_engineered_features(cls, weather_data: Dict[str, List], 
                                  event_duration: float = 1.0) -> Dict[str, List[float]]:
        """
        Compute all engineered features from weather data
        
        Args:
            weather_data: Dictionary with weather field lists (60 days each)
            event_duration: Duration of event in days for intensity calculations
            
        Returns:
            Dictionary with engineered feature lists
        """
        # Validate data
        validation = cls.validate_weather_data(weather_data)
        if not validation['valid']:
            raise ValueError(f"Invalid weather data: {validation['errors']}")
        
        days_count = validation['days_count']
        if days_count is None or days_count == 0:
            raise ValueError("No weather data provided")
        
        # Compute global statistics
        stats = cls.compute_global_stats(weather_data)
        
        # Initialize feature lists
        features = {feature: [] for feature in cls.ENGINEERED_FEATURES}
        
        # Process each day
        for day_idx in range(days_count):
            # Extract daily values with safe conversion
            temp = cls.safe_float(weather_data.get('temperature_C', [None] * days_count)[day_idx])
            temp_max = cls.safe_float(weather_data.get('temperature_max_C', [None] * days_count)[day_idx])
            temp_min = cls.safe_float(weather_data.get('temperature_min_C', [None] * days_count)[day_idx])
            humidity = cls.safe_float(weather_data.get('humidity_perc', [None] * days_count)[day_idx])
            spec_humidity = cls.safe_float(weather_data.get('specific_humidity_g_kg', [None] * days_count)[day_idx])
            dew_point = cls.safe_float(weather_data.get('dew_point_C', [None] * days_count)[day_idx])
            wind = cls.safe_float(weather_data.get('wind_speed_mps', [None] * days_count)[day_idx])
            wind_10m = cls.safe_float(weather_data.get('wind_speed_10m_mps', [None] * days_count)[day_idx])
            precip = cls.safe_float(weather_data.get('precipitation_mm', [None] * days_count)[day_idx])
            pressure = cls.safe_float(weather_data.get('surface_pressure_hPa', [None] * days_count)[day_idx])
            sea_pressure = cls.safe_float(weather_data.get('sea_level_pressure_hPa', [None] * days_count)[day_idx])
            solar = cls.safe_float(weather_data.get('solar_radiation_wm2', [None] * days_count)[day_idx])
            cloud = cls.safe_float(weather_data.get('cloud_amount_perc', [None] * days_count)[day_idx])
            soil_wetness = cls.safe_float(weather_data.get('surface_soil_wetness_perc', [None] * days_count)[day_idx])
            wind_dir = cls.safe_float(weather_data.get('wind_direction_10m_degrees', [None] * days_count)[day_idx])
            evap = cls.safe_float(weather_data.get('evapotranspiration_wm2', [None] * days_count)[day_idx])
            root_moisture = cls.safe_float(weather_data.get('root_zone_soil_moisture_perc', [None] * days_count)[day_idx])
            
            # Compute engineered features with NaN handling
            feature_values = cls._compute_daily_features(
                temp, temp_max, temp_min, humidity, spec_humidity, dew_point,
                wind, wind_10m, precip, pressure, sea_pressure, solar, cloud,
                soil_wetness, wind_dir, evap, root_moisture, event_duration, stats
            )
            
            # Add computed values to feature lists
            for feature_name, value in zip(cls.ENGINEERED_FEATURES, feature_values):
                features[feature_name].append(value)
        
        return features
    
    @classmethod 
    def _compute_daily_features(cls, temp, temp_max, temp_min, humidity, spec_humidity,
                              dew_point, wind, wind_10m, precip, pressure, sea_pressure,
                              solar, cloud, soil_wetness, wind_dir, evap, root_moisture,
                              event_duration, stats):
        """Compute engineered features for a single day with proper NaN handling.
        
        FIXED (v3): temp_normalized now uses daily T2M_MIN/T2M_MAX (matching
        step7 training pipeline where loop variables shadow global ones).
        drought_indicator also uses daily temp_max (same shadowing effect).
        """
        
        def safe_calc(func, *args, default=np.nan):
            """Safely execute calculation, return NaN if any input is NaN"""
            try:
                if any(np.isnan(arg) if isinstance(arg, (int, float)) else False for arg in args):
                    return default
                return func(*args)
            except (ZeroDivisionError, ValueError, TypeError):
                return default
        
        # 1. Temperature Normalization
        # FIXED: Uses daily T2M_MIN and T2M_MAX (not global min/max of T2M)
        # In step7, the loop variables `temp_min` and `temp_max` shadow the globals,
        # so temp_normalized = (T2M - T2M_MIN_daily) / (T2M_MAX_daily - T2M_MIN_daily)
        temp_normalized = safe_calc(
            lambda: (temp - temp_min) / (temp_max - temp_min) 
            if temp_max != temp_min else 0
        )
        
        # 2. Temperature Range (diurnal)
        temp_range = safe_calc(lambda: temp_max - temp_min)
        
        # 3. Discomfort Index (THI)
        discomfort_index = safe_calc(
            lambda: temp - 0.55 * (1 - 0.01 * humidity) * (temp - 14.5)
        )
        
        # 4. Heat Index
        heat_index = safe_calc(lambda: cls._calculate_heat_index(temp, humidity))
        
        # 5. Wind-Precipitation Interaction
        wind_precip_interaction = safe_calc(lambda: wind * precip)
        
        # 6. Solar Radiation to Temperature Ratio
        solar_temp_ratio = safe_calc(
            lambda: solar / (abs(temp) + 0.01) if abs(temp) + 0.01 > 1e-6 else 0
        )
        
        # 7. Pressure Anomaly (surface)
        pressure_anomaly = safe_calc(lambda: pressure - stats['pressure_mean'])
        
        # 8. High Precipitation Flag (>50mm threshold)
        high_precip_flag = safe_calc(lambda: int(precip > 50))
        
        # 9. Relative Humidity Adjusted for Temperature
        adjusted_humidity = safe_calc(lambda: humidity * (1 + (temp / 100)))
        
        # 10. Wind Chill Index
        wind_chill = safe_calc(lambda: cls._calculate_wind_chill(temp, wind))
        
        # 11. Solar Radiation Anomaly
        solar_anomaly = safe_calc(lambda: solar - stats['solar_mean'])
        
        # 12. Weather Severity Score (composite)
        weather_severity = safe_calc(lambda: (
            (temp_normalized if not np.isnan(temp_normalized) else 0) + 
            (precip / stats['precip_max'] if stats['precip_max'] != 0 else 0) + 
            (wind / stats['wind_max'] if stats['wind_max'] != 0 else 0) +
            (cloud / 100 if not np.isnan(cloud) else 0)
        ) / 4)
        
        # 13. Moisture Stress Index (evaporation vs precipitation)
        moisture_stress = safe_calc(
            lambda: (evap - precip) / (evap + precip + 0.01)
        )
        
        # 14. Evaporation Deficit
        evap_deficit = safe_calc(lambda: evap - stats['evap_mean'])
        
        # 15. Soil Saturation Index (combined soil moisture)
        soil_saturation = safe_calc(lambda: (soil_wetness + root_moisture) / 2)
        
        # 16. Atmospheric Instability (pressure difference + temp range)
        atm_instability = safe_calc(
            lambda: abs(sea_pressure - pressure) + (temp_range if not np.isnan(temp_range) else 0)
        )
        
        # 17. Drought Indicator (low precip + high temp + low soil moisture)
        # FIXED: Uses daily temp_max (T2M_MAX) not global stats['temp_max']
        # In step7, variable shadowing means the per-day T2M_MAX is used here
        drought_indicator = safe_calc(lambda: (
            (1 - precip / stats['precip_max']) * 
            ((temp - stats['temp_mean']) / max(abs(temp_max - stats['temp_mean']), 1)) * 
            (1 - (soil_saturation if not np.isnan(soil_saturation) else 0) / 100)
        ))
        
        # 18. Flood Risk Score (high precip + saturated soil + low evap)
        flood_risk = safe_calc(lambda: (
            (precip / stats['precip_max']) * 
            ((soil_saturation if not np.isnan(soil_saturation) else 0) / 100) * 
            (1 - evap / max(stats['evap_mean'] * 2, 1))
        ))
        
        # 19. Storm Intensity Index (wind + precip + pressure drop)  
        storm_intensity = safe_calc(lambda: (
            (wind_10m / stats['wind_max']) + 
            (precip / stats['precip_max']) + 
            (abs(pressure_anomaly if not np.isnan(pressure_anomaly) else 0) / 50)
        ))
        
        return [
            temp_normalized, temp_range, discomfort_index, heat_index,
            wind_precip_interaction, solar_temp_ratio, pressure_anomaly,
            high_precip_flag, adjusted_humidity, wind_chill, solar_anomaly,
            weather_severity, moisture_stress, evap_deficit, soil_saturation,
            atm_instability, drought_indicator, flood_risk, storm_intensity
        ]
    
    @staticmethod
    def _calculate_heat_index(temp, humidity):
        """Calculate heat index with NaN handling"""
        if np.isnan(temp) or np.isnan(humidity):
            return np.nan
            
        if temp >= 27 and humidity >= 40:
            return (-8.78469475556 + 1.61139411 * temp + 2.33854883889 * humidity +
                   -0.14611605 * temp * humidity + -0.012308094 * temp**2 +
                   -0.0164248277778 * humidity**2 + 0.002211732 * temp**2 * humidity +
                   0.00072546 * temp * humidity**2 + -0.000003582 * temp**2 * humidity**2)
        else:
            return temp
    
    @staticmethod
    def _calculate_wind_chill(temp, wind):
        """Calculate wind chill with NaN handling"""
        if np.isnan(temp) or np.isnan(wind):
            return np.nan
            
        if temp <= 10 and wind > 0:
            return (13.12 + 0.6215 * temp - 11.37 * np.power(wind, 0.16) + 
                   0.3965 * temp * np.power(wind, 0.16))
        else:
            return temp