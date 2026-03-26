"""
HazardGuard Disaster Prediction Model
Loads trained XGBoost model and predicts DISASTER vs NORMAL based on location coordinates
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import json
import ast
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class HazardGuardPredictionModel:
    """Model for HazardGuard disaster prediction using trained XGBoost classifier"""
    
    # Model files directory
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hazardguard', 'normal_vs_disaster')
    
    # Debug logging directory
    DEBUG_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hazardguard', 'debug_logs')
    
    # Feature columns expected by the model (based on training script)
    ARRAY_FEATURE_COLUMNS = [
        'temperature_C', 'humidity_%', 'wind_speed_mps', 'precipitation_mm',
        'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C', 'temperature_min_C',
        'specific_humidity_g_kg', 'dew_point_C', 'wind_speed_10m_mps', 'cloud_amount_%',
        'sea_level_pressure_hPa', 'surface_soil_wetness_%', 'wind_direction_10m_degrees', 'evapotranspiration_wm2',
        'root_zone_soil_moisture_%', 'temp_normalized', 'temp_range', 'discomfort_index',
        'heat_index', 'wind_precip_interaction', 'solar_temp_ratio', 'pressure_anomaly',
        'high_precip_flag', 'adjusted_humidity', 'wind_chill',
        'solar_radiation_anomaly', 'weather_severity_score', 'moisture_stress_index', 'evaporation_deficit',
        'soil_saturation_index', 'atmospheric_instability', 'drought_indicator', 'flood_risk_score',
        'storm_intensity_index'
    ]
    
    SCALAR_FEATURE_COLUMNS = [
        'soil_type', 'elevation_m', 'pop_density_persqkm', 'land_cover_class',
        'ndvi', 'annual_precip_mm', 'annual_mean_temp_c', 'mean_wind_speed_ms',
        'impervious_surface_pct'
    ]
    
    # HORIZON configuration for forecasting (1 day ahead prediction)
    HORIZON = 1
    FORECAST_DAYS = 60 - HORIZON  # Use first 59 days to predict day 60
    
    def __init__(self):
        """Initialize the HazardGuard prediction model"""
        model_root = os.getenv('MODEL_ROOT_PATH', '').strip()
        if model_root:
            base_hazardguard_dir = os.path.join(os.path.abspath(model_root), 'hazardguard')
        else:
            base_hazardguard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hazardguard')

        # Resolve model/debug directories from env-configured root when provided.
        self.MODEL_DIR = os.path.join(base_hazardguard_dir, 'normal_vs_disaster')
        self.DEBUG_LOG_DIR = os.path.join(base_hazardguard_dir, 'debug_logs')

        self.model = None
        self.feature_selector = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.imputation_values = None  # Add imputation values dictionary
        self.prediction_metadata = {}  # Metadata for current prediction (coordinates, dates, etc.)
        self.is_loaded = False
        
        # Statistics for monitoring
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'disaster_predictions': 0,
            'normal_predictions': 0,
            'avg_disaster_probability': 0.0,
            'model_load_time': None
        }
        
        # Create debug log directory if it doesn't exist
        os.makedirs(self.DEBUG_LOG_DIR, exist_ok=True)
        
        logger.info("HazardGuard prediction model initialized")
    
    def load_model_components(self) -> bool:
        """
        Load all trained model components
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Loading HazardGuard model components from {self.MODEL_DIR}")
            
            # Check if model directory exists
            if not os.path.exists(self.MODEL_DIR):
                logger.error(f"Model directory not found: {self.MODEL_DIR}")
                return False
            
            # Define component files
            model_files = {
                'model': 'normal_vs_disaster_xgboost_model.pkl',
                'feature_selector': 'normal_vs_disaster_feature_selector.pkl',
                'scaler': 'normal_vs_disaster_scaler.pkl',
                'label_encoder': 'normal_vs_disaster_label_encoder.pkl',
                'imputation_values': 'normal_vs_disaster_imputation_values.pkl',
                'metadata': 'normal_vs_disaster_model_metadata.json'
            }
            
            # Check if all required files exist (imputation_values is optional)
            missing_files = []
            for component, filename in model_files.items():
                if component == 'imputation_values':
                    # Optional - check later
                    continue
                filepath = os.path.join(self.MODEL_DIR, filename)
                if not os.path.exists(filepath):
                    missing_files.append(filename)
            
            if missing_files:
                logger.error(f"Missing model files: {missing_files}")
                return False
            
            # Load model components
            logger.info("Loading XGBoost model...")
            model_path = os.path.join(self.MODEL_DIR, model_files['model'])
            self.model = joblib.load(model_path)
            
            logger.info("Loading feature selector...")
            selector_path = os.path.join(self.MODEL_DIR, model_files['feature_selector'])
            self.feature_selector = joblib.load(selector_path)
            
            logger.info("Loading feature scaler...")
            scaler_path = os.path.join(self.MODEL_DIR, model_files['scaler'])
            self.scaler = joblib.load(scaler_path)
            
            logger.info("Loading label encoder...")
            encoder_path = os.path.join(self.MODEL_DIR, model_files['label_encoder'])
            self.label_encoder = joblib.load(encoder_path)
            
            logger.info("Loading imputation values...")
            imputation_path = os.path.join(self.MODEL_DIR, model_files['imputation_values'])
            if os.path.exists(imputation_path):
                self.imputation_values = joblib.load(imputation_path)
                logger.info(f"   Loaded imputation values for {len(self.imputation_values)} features")
                
                # Log training data statistics for raster features (for debugging)
                raster_features = ['soil_type', 'elevation_m', 'pop_density_persqkm', 'land_cover_class',
                                 'ndvi', 'annual_precip_mm', 'annual_mean_temp_c', 'mean_wind_speed_ms',
                                 'impervious_surface_pct']
                logger.info("   Training data imputation values for scalar raster features:")
                found_count = 0
                for feat in raster_features:
                    if feat in self.imputation_values:
                        logger.info(f"      {feat}: {self.imputation_values[feat]}")
                        found_count += 1
                    else:
                        logger.warning(f"      {feat}: NOT FOUND in imputation values")
                
                if found_count == 0:
                    logger.warning("   [CRITICAL] No scalar raster features in imputation file!")
                    logger.warning("   This means the model was trained with complete raster data (no missing values)")
                    logger.warning("   Current predictions use fallback imputation for missing raster values")
                
                # Log cloud_amount feature count
                cloud_features = [k for k in self.imputation_values.keys() if 'cloud_amount_%' in k]
                logger.info(f"   Found {len(cloud_features)} cloud_amount imputation values")
                # Check for cloud_amount features specifically
                cloud_keys = [k for k in self.imputation_values.keys() if 'cloud_amount' in k.lower()]
                logger.info(f"   Found {len(cloud_keys)} cloud_amount imputation values")
                if cloud_keys:
                    logger.debug(f"   Cloud_amount keys: {cloud_keys[:3]}...")
            else:
                logger.warning("   Imputation values file not found - will use fallback defaults")
                self.imputation_values = None
            
            logger.info("Loading model metadata...")
            metadata_path = os.path.join(self.MODEL_DIR, model_files['metadata'])
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_text = f.read()

                if metadata_text.lstrip().startswith('version https://git-lfs.github.com/spec/v1'):
                    logger.warning("   Metadata file is a Git LFS pointer; using fallback metadata")
                    self.metadata = {
                        'model_type': 'HazardGuard',
                        'algorithm': 'XGBoost',
                        'cv_accuracy': 0.0,
                        'n_features_selected': 0,
                        'metadata_source': 'fallback_lfs_pointer'
                    }
                else:
                    self.metadata = json.loads(metadata_text)
            except Exception as metadata_err:
                logger.warning(f"   Invalid metadata JSON ({metadata_err}); using fallback metadata")
                self.metadata = {
                    'model_type': 'HazardGuard',
                    'algorithm': 'XGBoost',
                    'cv_accuracy': 0.0,
                    'n_features_selected': 0,
                    'metadata_source': 'fallback_invalid_json'
                }
            
            # Calculate load time
            load_time = (datetime.now() - start_time).total_seconds()
            self.prediction_stats['model_load_time'] = load_time
            
            self.is_loaded = True
            
            logger.info("[SUCCESS] All model components loaded successfully!")
            logger.info(f"   Model type: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(f"   Algorithm: {self.metadata.get('algorithm', 'Unknown')}")
            logger.info(f"   CV Accuracy: {self.metadata.get('cv_accuracy', 0):.4f}")
            logger.info(f"   Features: {self.metadata.get('n_features_selected', 0)}")
            logger.info(f"   Load time: {load_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            return False
    
    def compute_stats_from_iterable(self, arr: List[float]) -> Dict[str, float]:
        """Compute statistics from a numeric array (same as training script)"""
        if len(arr) == 0:
            return {k: np.nan for k in ['mean','min','max','std','median','q25','q75','skew']}
        
        return {
            'mean': np.mean(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'std': np.std(arr) if len(arr) > 1 else 0.0,
            'median': np.median(arr),
            'q25': np.percentile(arr, 25),
            'q75': np.percentile(arr, 75),
            'skew': float(pd.Series(arr).skew()) if len(arr) > 2 else 0.0
        }
    
    def process_array_feature(self, values: List[float]) -> Dict[str, float]:
        """
        Process array feature for forecasting (use first FORECAST_DAYS values)
        
        Args:
            values: List of values (up to 60 days)
        
        Returns:
            Dictionary of computed statistics
        """
        try:
            # Handle None or empty lists
            if not values or len(values) == 0:
                return {k: np.nan for k in ['mean','min','max','std','median','q25','q75','skew']}
            
            # Filter out None, NaN, and empty values
            # Convert to float and filter valid values
            valid_values = []
            for x in values:
                try:
                    if x is not None and pd.notna(x) and str(x).strip() != '':
                        valid_values.append(float(x))
                except (ValueError, TypeError):
                    continue
            
            # If no valid values after filtering, return NaN
            if len(valid_values) == 0:
                logger.debug(f"No valid values in array feature after filtering None/NaN/blanks")
                return {k: np.nan for k in ['mean','min','max','std','median','q25','q75','skew']}
            
            # Convert to numpy array
            arr = np.array(valid_values)
            
            # For forecasting: use only first FORECAST_DAYS values
            if len(arr) > self.FORECAST_DAYS:
                arr = arr[:self.FORECAST_DAYS]
            
            return self.compute_stats_from_iterable(arr)
            
        except Exception as e:
            logger.error(f"Error processing array feature: {e}")
            return {k: np.nan for k in ['mean','min','max','std','median','q25','q75','skew']}
    
    def prepare_features(self, weather_data: Dict[str, List[float]], 
                        feature_data: Dict[str, List[float]], 
                        raster_data: Dict[str, float]) -> Optional[pd.DataFrame]:
        """
        Prepare features for prediction in the same format as training
        
        Args:
            weather_data: Weather time series data (60 days)
            feature_data: Engineered feature time series data (60 days)  
            raster_data: Raster data (single values)
        
        Returns:
            DataFrame with prepared features or None if error
        """
        try:
            logger.debug("Preparing features for prediction...")
            
            row_features = {}
            
            # Process array features (weather + engineered features)
            all_array_features = {**weather_data, **feature_data}
            
            for col in self.ARRAY_FEATURE_COLUMNS:
                if col in all_array_features:
                    values = all_array_features[col]
                    stats = self.process_array_feature(values)
                    
                    # Add statistics with column prefix
                    for stat, value in stats.items():
                        row_features[f"{col}_{stat}"] = value
                else:
                    # Missing feature - fill with NaN
                    logger.warning(f"Missing array feature: {col}")
                    for stat in ['mean','min','max','std','median','q25','q75','skew']:
                        row_features[f"{col}_{stat}"] = np.nan
            
            # Process scalar features (raster data)
            logger.info(f"[DEBUG] Processing {len(self.SCALAR_FEATURE_COLUMNS)} scalar features from raster data")
            logger.info(f"[DEBUG] Raster data keys: {list(raster_data.keys())}")
            
            raster_values_summary = {}
            raster_vs_training = {}  # Compare extracted vs training median
            
            for col in self.SCALAR_FEATURE_COLUMNS:
                if col in raster_data:
                    value = raster_data[col]
                    raster_values_summary[col] = value
                    
                    # Compare with training median if available
                    if self.imputation_values and col in self.imputation_values:
                        training_median = self.imputation_values[col]
                        if pd.notna(value) and value != -9999 and value != -9999.0:
                            diff_pct = ((value - training_median) / training_median * 100) if training_median != 0 else 0
                            raster_vs_training[col] = f"{value} (training: {training_median}, diff: {diff_pct:+.1f}%)"
                    
                    # Treat -9999 (NoData sentinel) as NaN
                    if pd.notna(value) and value != -9999 and value != -9999.0:
                        row_features[col] = value
                        logger.debug(f"Raster feature {col} = {value}")
                    else:
                        row_features[col] = np.nan
                        if value == -9999 or value == -9999.0:
                            logger.info(f"[DEBUG] Raster NoData value (-9999) for {col}, treating as NaN")
                else:
                    # Missing feature - fill with NaN
                    logger.warning(f"Missing scalar feature: {col}")
                    row_features[col] = np.nan
                    raster_values_summary[col] = "MISSING"
            
            logger.info(f"[DEBUG] All raster values: {raster_values_summary}")
            if raster_vs_training:
                logger.info(f"[DEBUG] Raster vs Training comparison: {raster_vs_training}")
            
            # Convert to DataFrame
            df = pd.DataFrame([row_features])
            
            # Handle missing values using EXACT training imputation values
            nan_counts_before = df.isnull().sum().sum()
            logger.info(f"[DEBUG] NaN values before imputation: {nan_counts_before}")
            
            # Log which features have NaN
            nan_features = df.columns[df.isnull().any()].tolist()
            if nan_features:
                logger.info(f"[DEBUG] Features with NaN: {nan_features}")
            
            imputed_features = {}
            imputed_from_training = 0
            imputed_from_fallback = 0
            
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    original_value = df[col].iloc[0] if len(df) > 0 else np.nan
                    # Use saved imputation values if available
                    if self.imputation_values and col in self.imputation_values:
                        fill_value = self.imputation_values[col]
                        df[col] = df[col].fillna(fill_value)
                        imputed_from_training += 1
                        logger.debug(f"Imputed {col} with training value: {fill_value}")
                    else:
                        imputed_from_fallback += 1
                        # Fallback: Use domain-specific defaults (matching working_live_predict.py)
                        if 'std' in col or 'skew' in col:
                            fill_value = 0
                            df[col] = df[col].fillna(fill_value)
                        # Scalar features (no training median, need reasonable defaults)
                        elif col == 'soil_type':
                            fill_value = 2
                            df[col] = df[col].fillna(fill_value)  # Default soil type
                        elif col == 'elevation_m':
                            fill_value = 300
                            df[col] = df[col].fillna(fill_value)  # Default elevation
                        elif col == 'pop_density_persqkm':
                            fill_value = 1000
                            df[col] = df[col].fillna(fill_value)  # Default population density
                        elif col == 'land_cover_class':
                            fill_value = 5
                            df[col] = df[col].fillna(fill_value)  # Default land cover
                        elif col == 'ndvi':
                            fill_value = 0.5
                            df[col] = df[col].fillna(fill_value)  # Default NDVI
                        elif col == 'annual_precip_mm':
                            fill_value = 800
                            df[col] = df[col].fillna(fill_value)  # Default annual precipitation
                        elif col == 'annual_mean_temp_c':
                            fill_value = 25
                            df[col] = df[col].fillna(fill_value)  # Default annual temperature
                        elif col == 'mean_wind_speed_ms':
                            fill_value = 3
                            df[col] = df[col].fillna(fill_value)  # Default wind speed
                        elif col == 'impervious_surface_pct':
                            fill_value = 10
                            df[col] = df[col].fillna(fill_value)  # Default impervious surface (10%)
                        # Array feature statistics
                        elif 'temperature' in col.lower():
                            fill_value = 20.0
                            df[col] = df[col].fillna(fill_value)
                        elif 'humidity' in col.lower() or 'cloud_amount' in col.lower():
                            fill_value = 50.0
                            df[col] = df[col].fillna(fill_value)
                        elif 'pressure' in col.lower():
                            fill_value = 1013.25
                            df[col] = df[col].fillna(fill_value)
                        else:
                            fill_value = 0.0
                            df[col] = df[col].fillna(fill_value)
                        
                        # Track imputed scalar features for debugging
                        if col in self.SCALAR_FEATURE_COLUMNS:
                            imputed_features[col] = fill_value
                        logger.warning(f"Imputed {col} with domain default (no training value available)")
            
            if imputed_features:
                logger.info(f"[DEBUG] Imputed scalar features: {imputed_features}")
            
            logger.info(f"[DEBUG] Imputation summary: {imputed_from_training} from training, {imputed_from_fallback} from fallback")
            
            # Final safety check
            if df.isnull().sum().sum() > 0:
                logger.warning(f"NaN values still present, filling with 0")
                df = df.fillna(0)
            
            logger.debug(f"Features prepared: {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def log_prediction_inputs(self, features_df: pd.DataFrame, weather_data: Dict, 
                             feature_data: Dict, raster_data: Dict, 
                             prediction: str, probability: Dict, metadata: Dict = None):
        """Log all prediction inputs to CSV for debugging"""
        try:
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'disaster_probability': probability.get('disaster', 0),
                'normal_probability': probability.get('normal', 0)
            }
            
            # Add metadata if provided (coordinates, dates, etc.)
            if metadata:
                for key, value in metadata.items():
                    log_entry[f'meta_{key}'] = value
            
            # Add all feature values (expanded features used by model)
            if features_df is not None:
                for col in features_df.columns:
                    log_entry[col] = features_df[col].iloc[0]
            
            # Add raw weather data arrays (60 values per column)
            for key, values in weather_data.items():
                if isinstance(values, list):
                    # Store the full array as a string representation
                    log_entry[key] = str(values) if len(values) > 0 else "[]"
                    # Also store mean for quick reference
                    if len(values) > 0:
                        valid_values = [v for v in values if pd.notna(v)]
                        if valid_values:
                            log_entry[f'{key}_mean'] = np.mean(valid_values)
            
            # Add raw engineered feature data arrays (60 values per column)
            for key, values in feature_data.items():
                if isinstance(values, list):
                    # Store the full array as a string representation
                    log_entry[key] = str(values) if len(values) > 0 else "[]"
                    # Also store mean for quick reference
                    if len(values) > 0:
                        valid_values = [v for v in values if pd.notna(v)]
                        if valid_values:
                            log_entry[f'{key}_mean'] = np.mean(valid_values)
            
            # Add raster data (scalar features)
            for key, value in raster_data.items():
                log_entry[key] = value
            
            # Convert to DataFrame and append to CSV
            log_df = pd.DataFrame([log_entry])
            log_file = os.path.join(self.DEBUG_LOG_DIR, 'prediction_inputs_log.csv')
            
            # Append to existing file or create new
            if os.path.exists(log_file):
                log_df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_file, index=False)
            
            logger.debug(f"Logged prediction inputs to {log_file}")
            
        except Exception as e:
            logger.error(f"Error logging prediction inputs: {e}")
    
    def predict_disaster(self, weather_data: Dict[str, List[float]], 
                        feature_data: Dict[str, List[float]], 
                        raster_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict disaster probability for given location data
        
        Args:
            weather_data: Pre-disaster weather time series 
            feature_data: Pre-disaster engineered features time series
            raster_data: Location raster data
        
        Returns:
            Prediction results dictionary
        """
        start_time = datetime.now()
        
        try:
            self.prediction_stats['total_predictions'] += 1
            
            # Check if model is loaded
            if not self.is_loaded or self.model is None:
                logger.error("Model not loaded. Call load_model_components() first.")
                self.prediction_stats['failed_predictions'] += 1
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'prediction': None,
                    'probability': None,
                    'confidence': None
                }
            
            # Prepare features
            features_df = self.prepare_features(weather_data, feature_data, raster_data)
            
            if features_df is None:
                self.prediction_stats['failed_predictions'] += 1
                return {
                    'success': False,
                    'error': 'Feature preparation failed',
                    'prediction': None,
                    'probability': None,
                    'confidence': None
                }
            
            # Apply feature selection
            logger.debug("Applying feature selection...")
            features_selected = self.feature_selector.transform(features_df)
            
            # Apply scaling
            logger.debug("Applying feature scaling...")
            features_scaled = self.scaler.transform(features_selected)
            
            # Make prediction
            logger.debug("Making prediction...")
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Debug: Log raw model output
            logger.info(f"[DEBUG] Raw prediction_encoded: {prediction_encoded}")
            logger.info(f"[DEBUG] Raw probabilities: {probabilities}")
            logger.info(f"[DEBUG] Label encoder classes: {self.label_encoder.classes_}")
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get probability for disaster class
            disaster_idx = list(self.label_encoder.classes_).index('Disaster') if 'Disaster' in self.label_encoder.classes_ else 1
            disaster_probability = probabilities[disaster_idx]
            normal_probability = probabilities[1 - disaster_idx]
            
            logger.info(f"[DEBUG] disaster_idx={disaster_idx}, disaster_prob={disaster_probability:.4f}, normal_prob={normal_probability:.4f}")
            logger.info(f"[DEBUG] Decoded prediction: {prediction}")
            
            # Calculate confidence (difference between max and second max probability)
            confidence = abs(disaster_probability - normal_probability)
            
            # Update statistics
            self.prediction_stats['successful_predictions'] += 1
            if prediction == 'Disaster':
                self.prediction_stats['disaster_predictions'] += 1
            else:
                self.prediction_stats['normal_predictions'] += 1
            
            # Update average disaster probability
            total_successful = self.prediction_stats['successful_predictions']
            current_avg = self.prediction_stats['avg_disaster_probability']
            self.prediction_stats['avg_disaster_probability'] = (
                (current_avg * (total_successful - 1) + disaster_probability) / total_successful
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'prediction': prediction,
                'probability': {
                    'disaster': float(disaster_probability),
                    'normal': float(normal_probability)
                },
                'confidence': float(confidence),
                'processing_time_seconds': processing_time,
                'metadata': {
                    'features_used': len(features_df.columns),
                    'features_selected': features_selected.shape[1],
                    'model_type': self.metadata.get('algorithm', 'XGBoost'),
                    'forecast_horizon_days': self.HORIZON,
                    'prediction_timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"[SUCCESS] Prediction successful: {prediction} (probability: {disaster_probability:.4f}, confidence: {confidence:.4f})")
            
            # Log prediction inputs for debugging
            self.log_prediction_inputs(
                features_df=features_df,
                weather_data=weather_data,
                feature_data=feature_data,
                raster_data=raster_data,
                prediction=prediction,
                probability=result['probability'],
                metadata=self.prediction_metadata
            )
            
            return result
            
        except Exception as e:
            self.prediction_stats['failed_predictions'] += 1
            logger.error(f"Error making prediction: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}",
                'prediction': None,
                'probability': None,
                'confidence': None,
                'processing_time_seconds': processing_time
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'is_loaded': self.is_loaded,
            'model_metadata': self.metadata if self.metadata else {},
            'feature_counts': {
                'array_features': len(self.ARRAY_FEATURE_COLUMNS),
                'scalar_features': len(self.SCALAR_FEATURE_COLUMNS),
                'total_expanded': len(self.ARRAY_FEATURE_COLUMNS) * 8 + len(self.SCALAR_FEATURE_COLUMNS)
            },
            'forecasting': {
                'horizon_days': self.HORIZON,
                'forecast_input_days': self.FORECAST_DAYS
            },
            'prediction_statistics': self.prediction_stats.copy()
        }
    
    def reset_statistics(self) -> None:
        """Reset prediction statistics"""
        self.prediction_stats.update({
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'disaster_predictions': 0,
            'normal_predictions': 0,
            'avg_disaster_probability': 0.0
        })
        logger.info("Prediction statistics reset")