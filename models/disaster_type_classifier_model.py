"""
Disaster Type Classifier Model
Multi-stage binary classification to identify specific disaster types
Runs 4 binary classifiers: Storm, Flood, Drought, Mass Movement (Landslide)

Each classifier is trained as "NO_<type> vs <type>" — i.e., the negative class
includes ALL other disaster types + Normal, not just Normal.
This makes them more robust one-vs-rest classifiers.

Models are XGBoost binary classifiers loaded from .joblib pipeline files containing:
  - 'model': XGBClassifier
  - 'scaler': StandardScaler
  - 'selector': SelectKBest (top 90% of 297 features)
  - 'selected_features': list of feature names after selection
  - 'target_disaster': positive class name
  - 'negative_label': negative class name (e.g. 'NO_Drought')
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DisasterTypeClassifierModel:
    """Model for classifying specific disaster types after disaster is detected"""
    
    def __init__(self):
        """Initialize disaster type classifier"""
        model_root = os.getenv('MODEL_ROOT_PATH', '').strip()
        if model_root:
            self.MODEL_BASE_DIR = os.path.join(os.path.abspath(model_root), 'hazardguard')
        else:
            self.MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hazardguard')
        
        # Define model paths for each disaster type (new binary NO_X vs X models)
        self.model_paths = {
            'Storm': {
                'pipeline': os.path.join(self.MODEL_BASE_DIR, 'binary_storm', 'binary_NOstorm_storm_pipeline.joblib'),
                'config': os.path.join(self.MODEL_BASE_DIR, 'binary_storm', 'comprehensive_model_config.json')
            },
            'Flood': {
                'pipeline': os.path.join(self.MODEL_BASE_DIR, 'binary_flood', 'binary_NOflood_flood_pipeline.joblib'),
                'config': os.path.join(self.MODEL_BASE_DIR, 'binary_flood', 'comprehensive_model_config.json')
            },
            'Drought': {
                'pipeline': os.path.join(self.MODEL_BASE_DIR, 'binary_drought', 'binary_NOdrought_drought_pipeline.joblib'),
                'config': os.path.join(self.MODEL_BASE_DIR, 'binary_drought', 'comprehensive_model_config.json')
            },
            'Landslide': {  # Mass Movement
                'pipeline': os.path.join(self.MODEL_BASE_DIR, 'binary_landslide', 'binary_NOmassmovement_massmovement_pipeline.joblib'),
                'config': os.path.join(self.MODEL_BASE_DIR, 'binary_landslide', 'comprehensive_model_config.json')
            }
        }
        
        # Model pipelines (loaded on demand or at initialization)
        self.models = {}
        self.models_loaded = False
        
        logger.info("Disaster type classifier model initialized")
    
    def load_models(self) -> bool:
        """Load all 4 binary disaster type classifier pipelines"""
        try:
            logger.info("Loading disaster type classification models...")
            
            for disaster_type, paths in self.model_paths.items():
                pipeline_path = paths['pipeline']
                
                # Check if pipeline file exists
                if not os.path.exists(pipeline_path):
                    logger.error(f"Missing {disaster_type} pipeline at {pipeline_path}")
                    return False
                
                # Load pipeline components (selector, scaler, model are stored separately)
                logger.info(f"  Loading {disaster_type} classifier...")
                loaded_data = joblib.load(pipeline_path)
                
                # Stored as dict with separate components: model, scaler, selector
                if isinstance(loaded_data, dict) and 'model' in loaded_data:
                    self.models[disaster_type] = loaded_data
                    logger.info(f"    [OK] {disaster_type} classifier loaded (selector + scaler + XGBoost)")
                else:
                    logger.error(f"{disaster_type}: Unexpected format - {type(loaded_data)}")
                    return False
            
            self.models_loaded = True
            logger.info(f"[SUCCESS] All {len(self.models)} disaster type classifiers loaded!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading disaster type models: {e}")
            self.models_loaded = False
            return False
    
    # HORIZON configuration matching training scripts
    HORIZON = 1
    FORECAST_DAYS = 60 - HORIZON  # Use first 59 days (same as training)
    
    def prepare_features_for_binary_models(self, weather_data: Dict[str, Any], feature_data: Dict[str, Any],
                                          raster_data: Dict[str, Any], lat: float, lon: float, reference_date: str) -> pd.DataFrame:
        """
        Prepare features for binary disaster type classifiers
        All 4 models (Storm/Flood/Drought/Landslide) use the same 36 array + 9 scalar features
        Arrays are truncated to first FORECAST_DAYS (59) values to match training HORIZON=1
        """
        try:
            # Flood, Drought, Landslide models use these 36 array features
            ARRAY_FEATURE_COLUMNS = [
                # Basic weather from NASA POWER (17 fields)
                'temperature_C', 'humidity_%', 'wind_speed_mps', 'precipitation_mm',
                'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C', 'temperature_min_C',
                'specific_humidity_g_kg', 'dew_point_C', 'wind_speed_10m_mps', 'cloud_amount_%',
                'sea_level_pressure_hPa', 'surface_soil_wetness_%', 'wind_direction_10m_degrees',
                'evapotranspiration_wm2', 'root_zone_soil_moisture_%',
                # Engineered features (19 fields)
                'temp_normalized', 'temp_range', 'discomfort_index', 'heat_index',
                'wind_precip_interaction', 'solar_temp_ratio', 'pressure_anomaly',
                'high_precip_flag', 'adjusted_humidity', 'wind_chill',
                'solar_radiation_anomaly', 'weather_severity_score',
                'moisture_stress_index', 'evaporation_deficit', 'soil_saturation_index',
                'atmospheric_instability', 'drought_indicator', 'flood_risk_score', 'storm_intensity_index'
            ]
            
            # Scalar features (9 raster features)
            SCALAR_FEATURE_COLUMNS = [
                'soil_type', 'elevation_m', 'pop_density_persqkm', 'land_cover_class',
                'ndvi', 'annual_precip_mm', 'annual_mean_temp_c', 'mean_wind_speed_ms',
                'impervious_surface_pct'
            ]
            
            row_features = {}
            
            # Parse reference date
            from datetime import datetime
            dt = datetime.strptime(reference_date, '%Y-%m-%d')
            
            # Process each array feature (expand to 8 statistics)
            missing_features = []
            for col in ARRAY_FEATURE_COLUMNS:
                # Check if it's a weather array
                if col in weather_data and isinstance(weather_data[col], list):
                    arr = weather_data[col][:self.FORECAST_DAYS]  # Truncate to first 59 days
                    if len(arr) > 0:
                        stats = self._compute_stats(arr)
                        for stat_name, stat_value in stats.items():
                            row_features[f"{col}_{stat_name}"] = stat_value
                    else:
                        for stat in ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']:
                            row_features[f"{col}_{stat}"] = np.nan
                
                # Check in engineered feature_data
                elif col in feature_data and isinstance(feature_data[col], list):
                    arr = feature_data[col][:self.FORECAST_DAYS]  # Truncate to first 59 days
                    if len(arr) > 0:
                        stats = self._compute_stats(arr)
                        for stat_name, stat_value in stats.items():
                            row_features[f"{col}_{stat_name}"] = stat_value
                    else:
                        for stat in ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']:
                            row_features[f"{col}_{stat}"] = np.nan
                
                # Missing array feature
                else:
                    missing_features.append(col)
                    for stat in ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']:
                        row_features[f"{col}_{stat}"] = np.nan
            
            if missing_features:
                logger.warning(f"Missing array features (will use NaN): {missing_features}")
            
            # Add scalar features directly (no statistics expansion)
            for col in SCALAR_FEATURE_COLUMNS:
                if col in raster_data:
                    value = raster_data[col]
                    if value == -9999 or value == -9999.0:
                        row_features[col] = np.nan
                    else:
                        row_features[col] = value
                else:
                    row_features[col] = np.nan
            
            # Convert to DataFrame
            df = pd.DataFrame([row_features])
            
            logger.debug(f"Prepared {len(df.columns)} features for binary classifiers (expected: 36x8 + 9 = 297)")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features for binary models: {e}")
            raise
    
    def _compute_stats(self, arr: List[float]) -> Dict[str, float]:
        """Compute 8 statistics from array with robust NaN handling"""
        if not isinstance(arr, (list, np.ndarray)):
            return {k: np.nan for k in ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']}
        
        # Convert to numpy array and filter out NaN/None values
        arr_clean = np.array([x for x in arr if pd.notna(x)], dtype=float)
        
        if len(arr_clean) == 0:
            return {k: np.nan for k in ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']}
        
        try:
            return {
                'mean': float(np.mean(arr_clean)),
                'min': float(np.min(arr_clean)),
                'max': float(np.max(arr_clean)),
                'std': float(np.std(arr_clean)) if len(arr_clean) > 1 else 0.0,
                'median': float(np.median(arr_clean)),
                'q25': float(np.percentile(arr_clean, 25)),
                'q75': float(np.percentile(arr_clean, 75)),
                'skew': float(pd.Series(arr_clean).skew()) if len(arr_clean) > 2 else 0.0
            }
        except Exception as e:
            logger.warning(f"Error computing stats: {e}")
            return {k: np.nan for k in ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']}
    
    def predict_disaster_types(self, weather_data: Dict[str, Any], feature_data: Dict[str, Any],
                               raster_data: Dict[str, Any], lat: float, lon: float, reference_date: str) -> Dict[str, Any]:
        """
        Run all 4 binary classifiers to determine which disaster types are predicted
        
        Returns:
            {
                'disaster_types': ['Storm', 'Flood'],  # List of predicted disasters
                'probabilities': {
                    'Storm': 0.85,
                    'Flood': 0.72,
                    'Drought': 0.15,
                    'Landslide': 0.08
                },
                'confidence': 'high'  # Based on probability scores
            }
        """
        try:
            if not self.models_loaded:
                logger.warning("Models not loaded, loading now...")
                if not self.load_models():
                    raise Exception("Failed to load disaster type models")
            
            # Prepare features
            logger.info("[DISASTER_TYPE] Preparing features for binary classifiers...")
            features = self.prepare_features_for_binary_models(weather_data, feature_data, raster_data, lat, lon, reference_date)
            
            # Impute NaN values using SAME logic as training
            # Training code: std/skew columns → 0, others → median
            nan_count = features.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"[DISASTER_TYPE] Found {nan_count} NaN values, imputing with training logic...")
                for col in features.columns:
                    if features[col].isnull().sum() > 0:
                        if 'std' in col or 'skew' in col:
                            features[col] = features[col].fillna(0)
                        else:
                            features[col] = features[col].fillna(features[col].median())
                
                # Replace inf/-inf with NaN then fill with median
                features = features.replace([np.inf, -np.inf], np.nan)
                for col in features.columns:
                    if features[col].isnull().sum() > 0:
                        features[col] = features[col].fillna(features[col].median())
                
                # Final fallback: any remaining NaN → 0
                features = features.fillna(0)
                logger.info(f"[DISASTER_TYPE] Imputation complete, remaining NaNs: {features.isna().sum().sum()}")
            
            # Run all 4 binary classifiers
            logger.info("[DISASTER_TYPE] Running binary disaster classifiers...")
            predictions = {}
            probabilities = {}
            
            for disaster_type, model_components in self.models.items():
                try:
                    # Each model is stored as dict with: selector, scaler, model
                    selector = model_components['selector']
                    scaler = model_components['scaler']
                    model = model_components['model']
                    
                    # Apply pipeline steps manually: selector → scaler → model
                    features_selected = selector.transform(features)
                    features_scaled = scaler.transform(features_selected)
                    
                    # Predict using XGBoost model
                    prediction = model.predict(features_scaled)[0]
                    proba = model.predict_proba(features_scaled)[0]
                    
                    # Get probability for disaster class (index 1 typically)
                    disaster_prob = proba[1] if len(proba) > 1 else proba[0]
                    
                    predictions[disaster_type] = prediction
                    probabilities[disaster_type] = float(disaster_prob)
                    
                    logger.info(f"  [{disaster_type}] Prediction={prediction}, Probability={disaster_prob:.4f}")
                    
                except Exception as model_error:
                    logger.error(f"Error predicting {disaster_type}: {model_error}")
                    predictions[disaster_type] = 0
                    probabilities[disaster_type] = 0.0
            
            # Determine which disasters are predicted (threshold: 0.5)
            predicted_disasters = [dt for dt, pred in predictions.items() if pred == 1]
            
            # Calculate overall confidence
            avg_prob = np.mean(list(probabilities.values()))
            confidence = 'high' if avg_prob > 0.7 else 'medium' if avg_prob > 0.5 else 'low'
            
            result = {
                'disaster_types': predicted_disasters,
                'probabilities': probabilities,
                'confidence': confidence,
                'details': f"Detected {len(predicted_disasters)} disaster type(s)" if predicted_disasters else "No specific disaster type detected"
            }
            
            logger.info(f"[DISASTER_TYPE] Predicted disasters: {predicted_disasters}")
            return result
            
        except Exception as e:
            logger.error(f"Error in disaster type prediction: {e}")
            raise
