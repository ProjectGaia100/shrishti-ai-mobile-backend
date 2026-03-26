"""
Post-Disaster Feature Engineering Model for HazardGuard System
Creates 19 advanced features from 60-day post-disaster weather data
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PostDisasterFeatureEngineeringModel:
    """Model for creating post-disaster features from weather time series data"""
    
    # Post-disaster weather variables expected as input (17 total)
    POST_WEATHER_VARIABLES = [
        'POST_temperature_C',
        'POST_humidity_%', 
        'POST_wind_speed_mps',
        'POST_precipitation_mm',
        'POST_surface_pressure_hPa',
        'POST_solar_radiation_wm2',
        'POST_temperature_max_C',
        'POST_temperature_min_C', 
        'POST_specific_humidity_g_kg',
        'POST_dew_point_C',
        'POST_wind_speed_10m_mps',
        'POST_cloud_amount_%',
        'POST_sea_level_pressure_hPa',
        'POST_surface_soil_wetness_%',
        'POST_wind_direction_10m_degrees',
        'POST_evapotranspiration_wm2',
        'POST_root_zone_soil_moisture_%'
    ]
    
    # Post-disaster engineered features (19 total)
    POST_FEATURE_VARIABLES = [
        'POST_temp_normalized',
        'POST_temp_range',
        'POST_discomfort_index',
        'POST_heat_index',
        'POST_wind_precip_interaction',
        'POST_solar_temp_ratio',
        'POST_pressure_anomaly',
        'POST_high_precip_flag',
        'POST_adjusted_humidity',
        'POST_wind_chill',
        'POST_solar_radiation_anomaly',
        'POST_weather_severity_score',
        'POST_moisture_stress_index',
        'POST_evaporation_deficit',
        'POST_soil_saturation_index',
        'POST_atmospheric_instability',
        'POST_drought_indicator',
        'POST_flood_risk_score',
        'POST_storm_intensity_index'
    ]
    
    def __init__(self, days_count: int = 60):
        """
        Initialize post-disaster feature engineering model
        
        Args:
            days_count: Number of days in time series (default: 60)
        """
        self.days_count = days_count
        self.global_stats = {}
        self.processing_stats = {
            'total_processed': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'nan_count': 0
        }
        
        logger.info(f"Initialized PostDisasterFeatureEngineeringModel: {days_count} days, {len(self.POST_FEATURE_VARIABLES)} features")
    
    def safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, handling NaN properly"""
        try:
            if pd.isna(value) or value is None:
                return np.nan
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def validate_weather_data(self, weather_data: Dict[str, List[float]]) -> Tuple[bool, str]:
        """Validate input weather data format"""
        try:
            # Check if all required variables are present
            missing_vars = []
            for var in self.POST_WEATHER_VARIABLES:
                if var not in weather_data:
                    missing_vars.append(var)
            
            if missing_vars:
                return False, f"Missing weather variables: {missing_vars}"
            
            # Check if all lists have correct length
            incorrect_lengths = []
            for var, values in weather_data.items():
                if var in self.POST_WEATHER_VARIABLES:
                    if not isinstance(values, list) or len(values) != self.days_count:
                        incorrect_lengths.append(f"{var}: {len(values) if isinstance(values, list) else 'not_list'}")
            
            if incorrect_lengths:
                return False, f"Incorrect list lengths (expected {self.days_count}): {incorrect_lengths}"
            
            return True, "Weather data validation successful"
            
        except Exception as e:
            logger.error(f"Error validating weather data: {e}")
            return False, f"Validation error: {str(e)}"
    
    def calculate_global_statistics(self, weather_datasets: List[Dict[str, List[float]]]) -> Dict[str, float]:
        """
        Calculate global statistics for normalization and anomaly detection
        
        Args:
            weather_datasets: List of weather data dictionaries for multiple coordinates
        
        Returns:
            Dictionary of global statistics
        """
        try:
            logger.info("Calculating global statistics for post-disaster feature engineering...")
            
            # Collect all values for each variable (flattened across all coordinates and days)
            all_values = {var: [] for var in self.POST_WEATHER_VARIABLES}
            
            for weather_data in weather_datasets:
                for var in self.POST_WEATHER_VARIABLES:
                    if var in weather_data and isinstance(weather_data[var], list):
                        for value in weather_data[var]:
                            float_val = self.safe_float(value, np.nan)
                            if not pd.isna(float_val):  # Only include non-NaN values for statistics
                                all_values[var].append(float_val)
            
            # Calculate statistics
            stats = {}
            
            # Temperature statistics
            temp_values = all_values['POST_temperature_C']
            stats['temp_min'] = float(np.min(temp_values)) if temp_values else 0.0
            stats['temp_max'] = float(np.max(temp_values)) if temp_values else 100.0
            stats['temp_mean'] = float(np.mean(temp_values)) if temp_values else 25.0
            
            temp_max_values = all_values['POST_temperature_max_C'] 
            stats['temp_max_mean'] = float(np.mean(temp_max_values)) if temp_max_values else 30.0
            
            # Pressure statistics
            pressure_values = all_values['POST_surface_pressure_hPa']
            stats['pressure_mean'] = float(np.mean(pressure_values)) if pressure_values else 1013.25
            
            sea_pressure_values = all_values['POST_sea_level_pressure_hPa']
            stats['sea_pressure_mean'] = float(np.mean(sea_pressure_values)) if sea_pressure_values else 1013.25
            
            # Solar radiation statistics
            solar_values = all_values['POST_solar_radiation_wm2']
            stats['solar_mean'] = float(np.mean(solar_values)) if solar_values else 200.0
            
            # Precipitation statistics
            precip_values = all_values['POST_precipitation_mm']
            stats['precip_max'] = float(np.max(precip_values)) if precip_values else 100.0
            
            # Wind statistics
            wind_values = all_values['POST_wind_speed_mps']
            stats['wind_max'] = float(np.max(wind_values)) if wind_values else 20.0
            
            # Evapotranspiration statistics
            evap_values = all_values['POST_evapotranspiration_wm2']
            stats['evap_mean'] = float(np.mean(evap_values)) if evap_values else 100.0
            
            # Store global statistics
            self.global_stats = stats
            
            logger.info(f"Global statistics calculated: {len(stats)} statistics computed")
            logger.debug(f"Global statistics: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating global statistics: {e}")
            return {
                'temp_min': 0.0, 'temp_max': 100.0, 'temp_mean': 25.0, 'temp_max_mean': 30.0,
                'pressure_mean': 1013.25, 'sea_pressure_mean': 1013.25, 'solar_mean': 200.0,
                'precip_max': 100.0, 'wind_max': 20.0, 'evap_mean': 100.0
            }
    
    def engineer_single_coordinate_features(self, weather_data: Dict[str, List[float]], global_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Engineer post-disaster features for a single coordinate
        
        Args:
            weather_data: Dictionary containing weather time series for all variables
            global_stats: Global statistics for normalization (optional)
        
        Returns:
            Dictionary containing engineered features and metadata
        """
        try:
            self.processing_stats['total_processed'] += 1
            
            # Validate input data
            is_valid, validation_message = self.validate_weather_data(weather_data)
            if not is_valid:
                self.processing_stats['failed_calculations'] += 1
                return {
                    'success': False,
                    'error': f"Weather data validation failed: {validation_message}",
                    'features': {feature: [np.nan] * self.days_count for feature in self.POST_FEATURE_VARIABLES}
                }
            
            # Use provided global stats or fallback to defaults
            stats = global_stats or self.global_stats or {
                'temp_min': 0.0, 'temp_max': 100.0, 'temp_mean': 25.0, 'temp_max_mean': 30.0,
                'pressure_mean': 1013.25, 'sea_pressure_mean': 1013.25, 'solar_mean': 200.0,
                'precip_max': 100.0, 'wind_max': 20.0, 'evap_mean': 100.0
            }
            
            # Initialize feature lists
            features = {feature: [] for feature in self.POST_FEATURE_VARIABLES}
            
            # Process each day
            for day in range(self.days_count):
                try:
                    # Extract daily values with safe conversion
                    temp = self.safe_float(weather_data['POST_temperature_C'][day], stats['temp_mean'])
                    temp_max = self.safe_float(weather_data['POST_temperature_max_C'][day], stats['temp_mean'] + 5)
                    temp_min = self.safe_float(weather_data['POST_temperature_min_C'][day], stats['temp_mean'] - 5)
                    humidity = self.safe_float(weather_data['POST_humidity_%'][day], 50.0)
                    spec_humidity = self.safe_float(weather_data['POST_specific_humidity_g_kg'][day], 10.0)
                    dew_point = self.safe_float(weather_data['POST_dew_point_C'][day], stats['temp_mean'] - 10)
                    wind = self.safe_float(weather_data['POST_wind_speed_mps'][day], 3.0)
                    wind_10m = self.safe_float(weather_data['POST_wind_speed_10m_mps'][day], 3.0)
                    precip = self.safe_float(weather_data['POST_precipitation_mm'][day], 0.0)
                    pressure = self.safe_float(weather_data['POST_surface_pressure_hPa'][day], stats['pressure_mean'])
                    sea_pressure = self.safe_float(weather_data['POST_sea_level_pressure_hPa'][day], stats['sea_pressure_mean'])
                    solar = self.safe_float(weather_data['POST_solar_radiation_wm2'][day], stats['solar_mean'])
                    cloud = self.safe_float(weather_data['POST_cloud_amount_%'][day], 50.0)
                    soil_wetness = self.safe_float(weather_data['POST_surface_soil_wetness_%'][day], 30.0)
                    wind_dir = self.safe_float(weather_data['POST_wind_direction_10m_degrees'][day], 180.0)
                    evap = self.safe_float(weather_data['POST_evapotranspiration_wm2'][day], stats['evap_mean'])
                    root_moisture = self.safe_float(weather_data['POST_root_zone_soil_moisture_%'][day], 30.0)
                    
                    # Count NaN values 
                    nan_count = sum(1 for val in [temp, temp_max, temp_min, humidity, spec_humidity, dew_point, 
                                                 wind, wind_10m, precip, pressure, sea_pressure, solar, cloud, 
                                                 soil_wetness, wind_dir, evap, root_moisture] if pd.isna(val))
                    if nan_count > 0:
                        self.processing_stats['nan_count'] += nan_count
                    
                    # 1. Temperature Normalization
                    if pd.isna(temp) or pd.isna(temp_min) or pd.isna(temp_max):
                        temp_normalized = np.nan
                    else:
                        temp_range_val = temp_max - temp_min if temp_max > temp_min else 1.0
                        temp_normalized = (temp - temp_min) / temp_range_val if temp_range_val > 0 else 0.0
                    features['POST_temp_normalized'].append(temp_normalized)
                    
                    # 2. Temperature Range (diurnal)
                    if pd.isna(temp_max) or pd.isna(temp_min):
                        temp_range = np.nan
                    else:
                        temp_range = temp_max - temp_min
                    features['POST_temp_range'].append(temp_range)
                    
                    # 3. Discomfort Index (THI)
                    if pd.isna(temp) or pd.isna(humidity):
                        discomfort_index = np.nan
                    else:
                        discomfort_index = temp - 0.55 * (1 - 0.01 * humidity) * (temp - 14.5)
                    features['POST_discomfort_index'].append(discomfort_index)
                    
                    # 4. Heat Index  
                    if pd.isna(temp) or pd.isna(humidity):
                        heat_index = np.nan
                    elif temp >= 27 and humidity >= 40:
                        heat_index = (-8.78469475556 + 1.61139411 * temp + 2.33854883889 * humidity +
                                     -0.14611605 * temp * humidity + -0.012308094 * temp**2 +
                                     -0.0164248277778 * humidity**2 + 0.002211732 * temp**2 * humidity +
                                     0.00072546 * temp * humidity**2 + -0.000003582 * temp**2 * humidity**2)
                    else:
                        heat_index = temp
                    features['POST_heat_index'].append(heat_index)
                    
                    # 5. Wind-Precipitation Interaction
                    if pd.isna(wind) or pd.isna(precip):
                        wind_precip_interaction = np.nan
                    else:
                        wind_precip_interaction = wind * precip
                    features['POST_wind_precip_interaction'].append(wind_precip_interaction)
                    
                    # 6. Solar Radiation to Temperature Ratio
                    if pd.isna(solar) or pd.isna(temp):
                        solar_temp_ratio = np.nan
                    else:
                        denominator = abs(temp) + 0.01
                        solar_temp_ratio = solar / denominator if denominator > 1e-6 else 0.0
                    features['POST_solar_temp_ratio'].append(solar_temp_ratio)
                    
                    # 7. Pressure Anomaly (surface)
                    if pd.isna(pressure):
                        pressure_anomaly = np.nan
                    else:
                        pressure_anomaly = pressure - stats['pressure_mean']
                    features['POST_pressure_anomaly'].append(pressure_anomaly)
                    
                    # 8. High Precipitation Flag (>50mm threshold)
                    if pd.isna(precip):
                        high_precip_flag = np.nan
                    else:
                        high_precip_flag = float(int(precip > 50))
                    features['POST_high_precip_flag'].append(high_precip_flag)
                    
                    # 9. Relative Humidity Adjusted for Temperature
                    if pd.isna(humidity) or pd.isna(temp):
                        adjusted_humidity = np.nan
                    else:
                        adjusted_humidity = humidity * (1 + (temp / 100))
                    features['POST_adjusted_humidity'].append(adjusted_humidity)
                    
                    # 10. Wind Chill Index
                    if pd.isna(temp) or pd.isna(wind):
                        wind_chill = np.nan
                    elif temp <= 10 and wind > 0:
                        wind_chill = (13.12 + 0.6215 * temp - 11.37 * np.power(wind, 0.16) + 
                                     0.3965 * temp * np.power(wind, 0.16))
                    else:
                        wind_chill = temp
                    features['POST_wind_chill'].append(wind_chill)
                    
                    # 11. Solar Radiation Anomaly
                    if pd.isna(solar):
                        solar_anomaly = np.nan
                    else:
                        solar_anomaly = solar - stats['solar_mean']
                    features['POST_solar_radiation_anomaly'].append(solar_anomaly)
                    
                    # 12. Weather Severity Score (composite)
                    if pd.isna(temp_normalized) or pd.isna(precip) or pd.isna(wind) or pd.isna(cloud):
                        weather_severity = np.nan
                    else:
                        precip_norm = precip / stats['precip_max'] if stats['precip_max'] > 0 else 0.0
                        wind_norm = wind / stats['wind_max'] if stats['wind_max'] > 0 else 0.0
                        cloud_norm = cloud / 100.0
                        weather_severity = (temp_normalized + precip_norm + wind_norm + cloud_norm) / 4.0
                    features['POST_weather_severity_score'].append(weather_severity)
                    
                    # 13. Moisture Stress Index (evaporation vs precipitation)
                    if pd.isna(evap) or pd.isna(precip):
                        moisture_stress = np.nan
                    else:
                        moisture_stress = (evap - precip) / (evap + precip + 0.01)
                    features['POST_moisture_stress_index'].append(moisture_stress)
                    
                    # 14. Evaporation Deficit
                    if pd.isna(evap):
                        evap_deficit = np.nan
                    else:
                        evap_deficit = evap - stats['evap_mean']
                    features['POST_evaporation_deficit'].append(evap_deficit)
                    
                    # 15. Soil Saturation Index (combined soil moisture)
                    if pd.isna(soil_wetness) or pd.isna(root_moisture):
                        soil_saturation = np.nan
                    else:
                        soil_saturation = (soil_wetness + root_moisture) / 2.0
                    features['POST_soil_saturation_index'].append(soil_saturation)
                    
                    # 16. Atmospheric Instability (pressure difference + temp range)
                    if pd.isna(sea_pressure) or pd.isna(pressure) or pd.isna(temp_range):
                        atm_instability = np.nan
                    else:
                        atm_instability = abs(sea_pressure - pressure) + temp_range
                    features['POST_atmospheric_instability'].append(atm_instability)
                    
                    # 17. Drought Indicator (low precip + high temp + low soil moisture)
                    if pd.isna(temp) or pd.isna(precip) or pd.isna(soil_saturation):
                        drought_indicator = np.nan
                    else:
                        temp_factor = (temp - stats['temp_mean']) / max(abs(stats['temp_max_mean'] - stats['temp_mean']), 1) if stats['temp_max_mean'] != stats['temp_mean'] else 0.0
                        drought_indicator = ((1 - precip / stats['precip_max']) * 
                                           max(0.0, temp_factor) * 
                                           (1 - soil_saturation / 100.0))
                    features['POST_drought_indicator'].append(drought_indicator)
                    
                    # 18. Flood Risk Score (high precip + saturated soil + low evap)
                    if pd.isna(precip) or pd.isna(soil_saturation) or pd.isna(evap):
                        flood_risk = np.nan
                    else:
                        precip_factor = precip / stats['precip_max'] if stats['precip_max'] > 0 else 0.0
                        soil_factor = soil_saturation / 100.0
                        evap_factor = 1.0 - evap / max(stats['evap_mean'] * 2, 1.0)
                        flood_risk = precip_factor * soil_factor * evap_factor
                    features['POST_flood_risk_score'].append(flood_risk)
                    
                    # 19. Storm Intensity Index (wind + precip + pressure drop)
                    if pd.isna(wind_10m) or pd.isna(precip) or pd.isna(pressure_anomaly):
                        storm_intensity = np.nan
                    else:
                        wind_factor = wind_10m / stats['wind_max'] if stats['wind_max'] > 0 else 0.0
                        precip_factor = precip / stats['precip_max'] if stats['precip_max'] > 0 else 0.0
                        pressure_factor = abs(pressure_anomaly) / 50.0
                        storm_intensity = wind_factor + precip_factor + pressure_factor
                    features['POST_storm_intensity_index'].append(storm_intensity)
                    
                except Exception as e:
                    logger.error(f"Error processing day {day}: {e}")
                    # Fill with NaN for this day across all features
                    for feature in self.POST_FEATURE_VARIABLES:
                        features[feature].append(np.nan)
            
            self.processing_stats['successful_calculations'] += 1
            
            return {
                'success': True,
                'features': features,
                'metadata': {
                    'days_processed': self.days_count,
                    'features_created': len(self.POST_FEATURE_VARIABLES),
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            self.processing_stats['failed_calculations'] += 1
            return {
                'success': False,
                'error': f"Feature engineering failed: {str(e)}",
                'features': {feature: [np.nan] * self.days_count for feature in self.POST_FEATURE_VARIABLES}
            }
    
    def engineer_batch_features(self, weather_datasets: List[Dict[str, List[float]]]) -> List[Dict[str, Any]]:
        """
        Engineer features for multiple coordinates with shared global statistics
        
        Args:
            weather_datasets: List of weather data dictionaries
        
        Returns:
            List of feature engineering results
        """
        try:
            logger.info(f"Engineering features for {len(weather_datasets)} coordinates")
            
            # Calculate global statistics across all datasets
            global_stats = self.calculate_global_statistics(weather_datasets)
            
            # Process each coordinate
            results = []
            for i, weather_data in enumerate(weather_datasets):
                logger.debug(f"Processing coordinate {i + 1}/{len(weather_datasets)}")
                result = self.engineer_single_coordinate_features(weather_data, global_stats)
                results.append(result)
            
            logger.info(f"Batch feature engineering completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch feature engineering error: {e}")
            return [
                {
                    'success': False,
                    'error': f"Batch processing failed: {str(e)}",
                    'features': {feature: [np.nan] * self.days_count for feature in self.POST_FEATURE_VARIABLES}
                }
                for _ in weather_datasets
            ]
    
    def get_feature_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Get descriptions of all engineered features"""
        return {
            'POST_temp_normalized': {
                'description': 'Normalized temperature based on daily range',
                'unit': 'ratio (0-1)',
                'calculation': '(temp - temp_min) / (temp_max - temp_min)'
            },
            'POST_temp_range': {
                'description': 'Diurnal temperature range',
                'unit': '°C',
                'calculation': 'temp_max - temp_min'
            },
            'POST_discomfort_index': {
                'description': 'Temperature-Humidity Index (THI)',
                'unit': '°C',
                'calculation': 'temp - 0.55 * (1 - 0.01 * humidity) * (temp - 14.5)'
            },
            'POST_heat_index': {
                'description': 'Apparent temperature combining temp and humidity',
                'unit': '°C',
                'calculation': 'Complex formula for temp>=27°C and humidity>=40%'
            },
            'POST_wind_precip_interaction': {
                'description': 'Wind-precipitation interaction term',
                'unit': 'mm·m/s',
                'calculation': 'wind_speed * precipitation'
            },
            'POST_solar_temp_ratio': {
                'description': 'Solar radiation efficiency relative to temperature',
                'unit': 'W/m²/°C',
                'calculation': 'solar_radiation / (|temperature| + 0.01)'
            },
            'POST_pressure_anomaly': {
                'description': 'Surface pressure deviation from global mean',
                'unit': 'hPa',
                'calculation': 'surface_pressure - global_pressure_mean'
            },
            'POST_high_precip_flag': {
                'description': 'Binary flag for heavy precipitation (>50mm)',
                'unit': 'binary',
                'calculation': '1 if precipitation > 50mm else 0'
            },
            'POST_adjusted_humidity': {
                'description': 'Relative humidity adjusted for temperature',
                'unit': '%',
                'calculation': 'humidity * (1 + temperature/100)'
            },
            'POST_wind_chill': {
                'description': 'Wind chill temperature for cold conditions',
                'unit': '°C',
                'calculation': 'Wind chill formula for temp<=10°C'
            },
            'POST_solar_radiation_anomaly': {
                'description': 'Solar radiation deviation from global mean',
                'unit': 'W/m²',
                'calculation': 'solar_radiation - global_solar_mean'
            },
            'POST_weather_severity_score': {
                'description': 'Composite weather severity index',
                'unit': 'ratio (0-1)',
                'calculation': 'Average of normalized temp, precip, wind, cloud metrics'
            },
            'POST_moisture_stress_index': {
                'description': 'Evapotranspiration vs precipitation balance',
                'unit': 'ratio (-1 to 1)',
                'calculation': '(evap - precip) / (evap + precip + 0.01)'
            },
            'POST_evaporation_deficit': {
                'description': 'Evapotranspiration deficit from global mean',
                'unit': 'W/m²',
                'calculation': 'evapotranspiration - global_evap_mean'
            },
            'POST_soil_saturation_index': {
                'description': 'Combined soil moisture indicator',
                'unit': '%',
                'calculation': '(surface_wetness + root_moisture) / 2'
            },
            'POST_atmospheric_instability': {
                'description': 'Atmospheric instability indicator',
                'unit': 'hPa + °C',
                'calculation': '|sea_pressure - surface_pressure| + temp_range'
            },
            'POST_drought_indicator': {
                'description': 'Composite drought risk index',
                'unit': 'ratio (0-1)',
                'calculation': 'Function of low precip, high temp, low soil moisture'
            },
            'POST_flood_risk_score': {
                'description': 'Composite flood risk index',
                'unit': 'ratio (0-1)',
                'calculation': 'Function of high precip, saturated soil, low evap'
            },
            'POST_storm_intensity_index': {
                'description': 'Composite storm intensity index',
                'unit': 'ratio',
                'calculation': 'Sum of normalized wind, precip, pressure anomaly'
            }
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_processed = self.processing_stats['total_processed']
        
        return {
            'total_coordinates_processed': total_processed,
            'successful_calculations': self.processing_stats['successful_calculations'],
            'failed_calculations': self.processing_stats['failed_calculations'],
            'success_rate': (self.processing_stats['successful_calculations'] / total_processed * 100) if total_processed > 0 else 0,
            'nan_values_encountered': self.processing_stats['nan_count'],
            'days_per_coordinate': self.days_count,
            'features_per_coordinate': len(self.POST_FEATURE_VARIABLES),
            'input_variables': len(self.POST_WEATHER_VARIABLES),
            'output_variables': len(self.POST_FEATURE_VARIABLES)
        }