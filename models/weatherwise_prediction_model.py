"""
WeatherWise Prediction Model
LSTM-based weather forecasting models for different disaster contexts
"""

import logging
import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# TensorFlow imported at startup so it loads once predictably.
# AttentionLayer is defined inside load_models() to avoid constructing
# Keras layers before a model context is ready.
import tensorflow as tf

logger = logging.getLogger(__name__)

class WeatherWisePredictionModel:
    """Model class for WeatherWise weather forecasting using LSTM models"""
    
    def __init__(self):
        """Initialize WeatherWise prediction model"""
        self.models = {}
        self.input_scalers = {}
        self.output_scalers = {}
        self.model_info = {}
        
        model_root = os.getenv('MODEL_ROOT_PATH', '').strip()
        if model_root:
            self.base_path = os.path.join(os.path.abspath(model_root), 'weatherwise')
        else:
            # Model paths - fallback to organized subfolders inside backend/server/models/weatherwise/
            self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weatherwise')

        logger.info(f"[WEATHERWISE_MODEL] Base path for models: {self.base_path}")
        
        self.model_paths = {
            'Normal': os.path.join(self.base_path, 'normal'),
            'Flood': os.path.join(self.base_path, 'flood'),
            'Drought': os.path.join(self.base_path, 'drought'),
            'Landslide': os.path.join(self.base_path, 'landslide'),
            'Storm': os.path.join(self.base_path, 'storm')
        }
        
        # Weather variables expected by LSTM models (36 input features: 17 raw + 19 engineered)
        self.weather_features = [
            'temperature_C', 'humidity_%', 'wind_speed_mps', 'precipitation_mm',
            'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C', 
            'temperature_min_C', 'specific_humidity_g_kg', 'dew_point_C', 
            'wind_speed_10m_mps', 'cloud_amount_%', 'sea_level_pressure_hPa', 
            'surface_soil_wetness_%', 'wind_direction_10m_degrees', 
            'evapotranspiration_wm2', 'root_zone_soil_moisture_%',
            'temp_normalized', 'temp_range', 'discomfort_index', 'heat_index',
            'wind_precip_interaction', 'solar_temp_ratio', 'pressure_anomaly',
            'high_precip_flag', 'adjusted_humidity', 'wind_chill',
            'solar_radiation_anomaly', 'weather_severity_score',
            'moisture_stress_index', 'evaporation_deficit', 'soil_saturation_index',
            'atmospheric_instability', 'drought_indicator', 'flood_risk_score',
            'storm_intensity_index'
        ]
        
        # Output weather variables (all 36 features the model predicts)
        self.forecast_variables = [
            'temperature_C', 'humidity_%', 'wind_speed_mps', 'precipitation_mm',
            'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C',
            'temperature_min_C', 'specific_humidity_g_kg', 'dew_point_C',
            'wind_speed_10m_mps', 'cloud_amount_%', 'sea_level_pressure_hPa',
            'surface_soil_wetness_%', 'wind_direction_10m_degrees',
            'evapotranspiration_wm2', 'root_zone_soil_moisture_%', 'temp_normalized',
            'temp_range', 'discomfort_index', 'heat_index',
            'wind_precip_interaction', 'solar_temp_ratio', 'pressure_anomaly',
            'high_precip_flag', 'adjusted_humidity', 'wind_chill',
            'solar_radiation_anomaly', 'weather_severity_score',
            'moisture_stress_index', 'evaporation_deficit', 'soil_saturation_index',
            'atmospheric_instability', 'drought_indicator', 'flood_risk_score',
            'storm_intensity_index'
        ]
        
        logger.info("WeatherWise prediction model initialized")
    
    def load_models(self) -> bool:
        """
        Load all LSTM weather prediction models
        
        Returns:
            bool: True if models loaded successfully
        """
        try:
            # AttentionLayer defined here so it is only constructed when models are loaded.
            class AttentionLayer(tf.keras.layers.Layer):
                """Bahdanau Attention mechanism for Seq2Seq model."""
                def __init__(self, units, **kwargs):
                    super(AttentionLayer, self).__init__(**kwargs)
                    self.units = units
                    self.W1 = tf.keras.layers.Dense(units)
                    self.W2 = tf.keras.layers.Dense(units)
                    self.V  = tf.keras.layers.Dense(1)

                def call(self, query, values):
                    query_with_time_axis = tf.expand_dims(query, 1)
                    score = self.V(tf.nn.tanh(
                        self.W1(query_with_time_axis) + self.W2(values)
                    ))
                    attention_weights = tf.nn.softmax(score, axis=1)
                    context_vector = attention_weights * values
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, attention_weights

                def get_config(self):
                    config = super().get_config()
                    config.update({"units": self.units})
                    return config

            logger.info("Loading WeatherWise LSTM models...")

            for disaster_type, model_path in self.model_paths.items():
                logger.info(f"[WEATHERWISE_MODEL] Checking {disaster_type} model at: {model_path}")
                
                if not os.path.exists(model_path):
                    logger.warning(f"Model path not found for {disaster_type}: {model_path}")
                    continue
                
                try:
                    # Load model components
                    model_file = os.path.join(model_path, 'best_model.keras')
                    input_scaler_file = os.path.join(model_path, 'input_scaler.joblib')
                    output_scaler_file = os.path.join(model_path, 'output_scaler.joblib')
                    info_file = os.path.join(model_path, 'model_info.json')
                    
                    logger.info(f"[WEATHERWISE_MODEL] Checking files for {disaster_type}:")
                    logger.info(f"  - Model file exists: {os.path.exists(model_file)}")
                    logger.info(f"  - Input scaler exists: {os.path.exists(input_scaler_file)}")
                    logger.info(f"  - Output scaler exists: {os.path.exists(output_scaler_file)}")
                    
                    if all(os.path.exists(f) for f in [model_file, input_scaler_file, output_scaler_file]):
                        # Load model with custom AttentionLayer
                        model = tf.keras.models.load_model(
                            model_file,
                            custom_objects={'AttentionLayer': AttentionLayer}
                        )
                        self.models[disaster_type] = model
                        
                        # Load scalers
                        self.input_scalers[disaster_type] = joblib.load(input_scaler_file)
                        self.output_scalers[disaster_type] = joblib.load(output_scaler_file)
                        
                        # Load model info if available
                        if os.path.exists(info_file):
                            try:
                                with open(info_file, 'r', encoding='utf-8') as f:
                                    info_text = f.read()
                                if info_text.lstrip().startswith('version https://git-lfs.github.com/spec/v1'):
                                    logger.warning(
                                        f"[WEATHERWISE_MODEL] model_info.json for {disaster_type} is a Git LFS pointer; using defaults"
                                    )
                                    self.model_info[disaster_type] = {'horizon_days': 60}
                                else:
                                    self.model_info[disaster_type] = json.loads(info_text)
                            except Exception as info_err:
                                logger.warning(
                                    f"[WEATHERWISE_MODEL] Invalid model_info.json for {disaster_type}: {info_err}; using defaults"
                                )
                                self.model_info[disaster_type] = {'horizon_days': 60}
                        else:
                            self.model_info[disaster_type] = {'horizon_days': 60}
                        
                        logger.info(f"[OK] Loaded {disaster_type} weather model")
                    else:
                        logger.warning(f"Missing model files for {disaster_type}")
                        
                except Exception as e:
                    logger.error(f"Failed to load {disaster_type} model: {e}")
                    continue
            
            if self.models:
                logger.info(f"[SUCCESS] Loaded {len(self.models)} WeatherWise models: {list(self.models.keys())}")
                return True
            else:
                logger.error("[ERROR] No WeatherWise models loaded")
                return False
                
        except Exception as e:
            logger.error(f"WeatherWise model loading error: {e}")
            return False
    
    def prepare_input_sequence(self, weather_data: Dict[str, List[float]], 
                             feature_data: Dict[str, List[float]]) -> np.ndarray:
        """
        Prepare input sequence for LSTM model from weather and feature data
        
        Args:
            weather_data: Weather time series data (59-60 days)
            feature_data: Engineered features data (59-60 days)
        
        Returns:
            np.ndarray: Prepared input sequence for LSTM
        """
        try:
            # Determine the actual length of data
            data_length = 60  # Default
            for var_name, var_data in weather_data.items():
                if var_data and isinstance(var_data, list):
                    data_length = len(var_data)
                    break
            
            logger.info(f"[WEATHERWISE] Preparing input sequence with {data_length} timesteps")
            
            # Combine weather and feature data - use ONLY the 36 features the model expects
            combined_data = {}
            
            for var in self.weather_features:
                # First check weather_data (raw weather variables)
                if var in weather_data and weather_data[var]:
                    data = weather_data[var]
                # Then check feature_data (engineered features)
                elif var in feature_data and feature_data[var]:
                    data = feature_data[var]
                else:
                    # Fill missing data with zeros
                    combined_data[var] = [0.0] * data_length
                    continue
                
                # Ensure correct length
                if len(data) < data_length:
                    data = data + [data[-1]] * (data_length - len(data))
                elif len(data) > data_length:
                    data = data[:data_length]
                combined_data[var] = data
            
            logger.info(f"[WEATHERWISE] Combined data has {len(combined_data)} features")
            
            # Convert to numpy array (timesteps, features)
            feature_names = sorted(combined_data.keys())  # Sort for consistency
            
            # Build array ensuring all same length and convert to float
            arrays = []
            for name in feature_names:
                arr = combined_data[name]
                if len(arr) != data_length:
                    logger.warning(f"Feature {name} has length {len(arr)}, expected {data_length}")
                    # Fix length mismatch
                    if len(arr) < data_length:
                        arr = arr + [0.0] * (data_length - len(arr))
                    else:
                        arr = arr[:data_length]
                
                # Convert to float and replace None/NaN with 0.0
                float_arr = []
                for val in arr:
                    if val is None:
                        float_arr.append(0.0)
                    elif isinstance(val, (int, float)):
                        if np.isnan(val) or np.isinf(val):
                            float_arr.append(0.0)
                        else:
                            float_arr.append(float(val))
                    else:
                        # Non-numeric, use 0.0
                        float_arr.append(0.0)
                arrays.append(float_arr)
            
            sequence_data = np.array(arrays, dtype=np.float32).T  # Transpose to (timesteps, features) with explicit float type
            
            # Reshape for LSTM: (1, timesteps, features)
            input_sequence = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
            
            logger.info(f"[WEATHERWISE] Prepared input sequence shape: {input_sequence.shape}")
            return input_sequence
            
        except Exception as e:
            logger.error(f"Input sequence preparation error: {e}")
            raise
    
    def predict_weather_forecast(self, weather_data: Dict[str, List[float]], 
                               feature_data: Dict[str, List[float]], 
                               disaster_type: str = 'Normal',
                               forecast_days: int = 60) -> Dict[str, Any]:
        """
        Generate weather forecast using LSTM model
        
        Args:
            weather_data: Historical weather data (60 days)
            feature_data: Engineered features data (60 days) 
            disaster_type: Type of disaster context for model selection
            forecast_days: Number of days to forecast (default 60)
        
        Returns:
            Dict containing forecast results
        """
        try:
            start_time = datetime.now()
            
            # Validate disaster type
            if disaster_type not in self.models:
                available_models = list(self.models.keys())
                if available_models:
                    disaster_type = available_models[0]  # Use first available model
                    logger.warning(f"Requested disaster type not available, using {disaster_type}")
                else:
                    return {'success': False, 'error': 'No models available'}
            
            # Get model components
            model = self.models[disaster_type]
            input_scaler = self.input_scalers[disaster_type]
            output_scaler = self.output_scalers[disaster_type]
            
            logger.info(f"[WEATHERWISE] Generating {forecast_days}-day forecast using {disaster_type} model")
            
            # Debug: Check input data quality
            logger.debug(f"[WEATHERWISE] Received weather_data with {len(weather_data)} variables")
            logger.debug(f"[WEATHERWISE] Received feature_data with {len(feature_data)} variables")
            
            # Check for NaN in weather_data
            weather_nan_vars = []
            for var, values in weather_data.items():
                if values and any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in values):
                    weather_nan_vars.append(var)
            if weather_nan_vars:
                logger.warning(f"[WEATHERWISE] Weather data contains NaN/Inf in variables: {weather_nan_vars}")
            
            # Check for NaN in feature_data  
            feature_nan_vars = []
            for var, values in feature_data.items():
                if values and any(isinstance(v, float) and (np.isnan(v) or np.isinf(v)) for v in values):
                    feature_nan_vars.append(var)
            if feature_nan_vars:
                logger.warning(f"[WEATHERWISE] Feature data contains NaN/Inf in variables: {feature_nan_vars}")
            
            # Prepare input sequence
            input_sequence = self.prepare_input_sequence(weather_data, feature_data)
            
            # Validate input sequence dtype and check for NaN
            if input_sequence.dtype == np.object_:
                logger.error(f"[WEATHERWISE] Input sequence has object dtype! Converting to float...")
                input_sequence = input_sequence.astype(np.float32)
            
            # Check input for NaN (now safe since we have numeric dtype)
            input_nan_count = np.isnan(input_sequence).sum()
            if input_nan_count > 0:
                logger.warning(f"[WEATHERWISE] Input sequence contains {input_nan_count} NaN values before scaling")
                logger.debug(f"[WEATHERWISE] Input stats - Min: {np.nanmin(input_sequence):.4f}, Max: {np.nanmax(input_sequence):.4f}, Mean: {np.nanmean(input_sequence):.4f}")
                # Replace NaN with 0 for model stability
                input_sequence = np.nan_to_num(input_sequence, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"[WEATHERWISE] Replaced NaN values with 0.0")
            else:
                logger.info(f"[WEATHERWISE] Input sequence is clean - Min: {np.min(input_sequence):.4f}, Max: {np.max(input_sequence):.4f}, Mean: {np.mean(input_sequence):.4f}")
            
            # Scale input data
            original_shape = input_sequence.shape
            input_flat = input_sequence.reshape(-1, input_sequence.shape[-1])
            input_scaled = input_scaler.transform(input_flat)
            input_scaled = input_scaled.reshape(original_shape)
            
            # Check scaled input for NaN
            scaled_nan_count = np.isnan(input_scaled).sum()
            if scaled_nan_count > 0:
                logger.error(f"[WEATHERWISE] Scaled input contains {scaled_nan_count} NaN values! Scaler may be corrupted.")
                logger.debug(f"[WEATHERWISE] Scaled stats - Min: {np.nanmin(input_scaled):.4f}, Max: {np.nanmax(input_scaled):.4f}")
            
            # Generate forecast
            forecast_scaled = model.predict(input_scaled, verbose=0)
            
            # Check model output for NaN
            model_output_nan = np.isnan(forecast_scaled).sum()
            if model_output_nan > 0:
                logger.error(f"[WEATHERWISE] Model output contains {model_output_nan} NaN values!")
                logger.debug(f"[WEATHERWISE] Model output stats - Min: {np.nanmin(forecast_scaled):.4f}, Max: {np.nanmax(forecast_scaled):.4f}")
            
            # Inverse scale forecast
            forecast_shape = forecast_scaled.shape
            forecast_flat = forecast_scaled.reshape(-1, forecast_scaled.shape[-1])
            forecast = output_scaler.inverse_transform(forecast_flat)
            forecast = forecast.reshape(forecast_shape)
            
            # Check final forecast for NaN
            final_nan_count = np.isnan(forecast).sum()
            if final_nan_count > 0:
                logger.error(f"[WEATHERWISE] Final forecast contains {final_nan_count} NaN values after inverse scaling!")
                logger.debug(f"[WEATHERWISE] Final forecast stats - Min: {np.nanmin(forecast):.4f}, Max: {np.nanmax(forecast):.4f}")
            else:
                logger.info(f"[WEATHERWISE] Forecast generated successfully - Min: {np.min(forecast):.4f}, Max: {np.max(forecast):.4f}")
            
            # Extract forecast for requested number of days
            forecast_data = forecast[0][:forecast_days]  # Take first batch, limit days
            
            # Format forecast results and handle NaN values
            forecast_dict = {}
            nan_count = 0
            for i, var in enumerate(self.forecast_variables):
                if i < forecast_data.shape[1]:
                    # Convert to list and replace NaN with None (JSON null)
                    values = forecast_data[:, i].tolist()
                    # Replace NaN and inf values with None for JSON serialization
                    cleaned_values = []
                    for v in values:
                        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                            cleaned_values.append(None)
                            nan_count += 1
                        else:
                            cleaned_values.append(float(v))
                    forecast_dict[var] = cleaned_values
                else:
                    forecast_dict[var] = [0.0] * forecast_days
            
            if nan_count > 0:
                logger.warning(f"[WEATHERWISE] Replaced {nan_count} NaN/Inf values with null in forecast")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate forecast dates
            base_date = datetime.now().date()
            forecast_dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                            for i in range(forecast_days)]
            
            return {
                'success': True,
                'model_type': disaster_type,
                'forecast_horizon_days': forecast_days,
                'forecast_dates': forecast_dates,
                'weather_forecast': forecast_dict,
                'forecast_variables': self.forecast_variables,
                'processing_time_seconds': processing_time,
                'model_info': self.model_info.get(disaster_type, {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"WeatherWise forecast error: {e}")
            return {
                'success': False,
                'error': f'Forecast generation failed: {str(e)}',
                'model_type': disaster_type,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available disaster context models"""
        return list(self.models.keys())
    
    def get_model_info(self, disaster_type: str = None) -> Dict[str, Any]:
        """Get information about loaded models"""
        if disaster_type and disaster_type in self.model_info:
            return self.model_info[disaster_type]
        return {
            'available_models': self.get_available_models(),
            'forecast_variables': self.forecast_variables,
            'input_features': len(self.weather_features) + 9,  # weather + engineered features
            'default_horizon_days': 60
        }