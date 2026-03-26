"""
GeoVision Fusion Prediction Model
===================================
Cross-stacked ensemble fusion pipeline that combines:
  1. LSTM MIMO model  → disaster probs (5), weather probs (5), temporal embeddings (128)
  2. Tree Ensemble    → disaster probs (5)  [RF + XGB + ET + LGBM + CB]
  3. CNN ResNet50     → disaster probs (5)  [OPTIONAL — requires satellite imagery]
  4. Fusion Meta-Learner → final disaster prediction + weather regime prediction

Output classes:
  Disaster: ['Drought', 'Flood', 'Landslide', 'Normal', 'Storm']  (alphabetical LabelEncoder order)
  Weather:  ['Cloudy', 'Dry', 'Humid', 'Rainy', 'Stormy']        (alphabetical LabelEncoder order)

The fusion meta-learner builds a feature vector by concatenating the intermediate
probability outputs and embeddings, then feeds it through an XGBoost/CatBoost/LogReg
meta-learner trained via 5-fold cross-validated stacking.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS (matching training scripts exactly)
# ════════════════════════════════════════════════════════════════════
MASK_VALUE = -999.0
SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 128
HORIZON = 1
FORECAST_DAYS = SEQUENCE_LENGTH - HORIZON  # 59

DISASTER_CLASSES = ['Drought', 'Flood', 'Landslide', 'Normal', 'Storm']  # Alphabetical (LabelEncoder)
WEATHER_REGIMES  = ['Cloudy', 'Dry', 'Humid', 'Rainy', 'Stormy']        # Alphabetical (LabelEncoder)

# Note: The fusion training script uses ['Normal','Flood','Storm','Drought','Landslide']
# but the LabelEncoder produces alphabetical order. The inference pipeline uses
# DISASTER_CLASSES = ['Normal','Flood','Storm','Drought','Landslide'] for display
# but internally the model predicts indices 0-4 in alphabetical order.
# We follow the alphabetical order as the ground truth from the LabelEncoder.

# 36 temporal features (17 raw meteorological + 19 engineered)
INPUT_FEATURES = [
    'temperature_C', 'humidity_%', 'wind_speed_mps', 'precipitation_mm',
    'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C', 'temperature_min_C',
    'specific_humidity_g_kg', 'dew_point_C', 'wind_speed_10m_mps', 'cloud_amount_%',
    'sea_level_pressure_hPa', 'surface_soil_wetness_%', 'wind_direction_10m_degrees',
    'evapotranspiration_wm2', 'root_zone_soil_moisture_%',
    'temp_normalized', 'temp_range', 'discomfort_index', 'heat_index',
    'wind_precip_interaction', 'solar_temp_ratio', 'pressure_anomaly',
    'high_precip_flag', 'adjusted_humidity', 'wind_chill',
    'solar_radiation_anomaly', 'weather_severity_score',
    'moisture_stress_index', 'evaporation_deficit', 'soil_saturation_index',
    'atmospheric_instability', 'drought_indicator', 'flood_risk_score', 'storm_intensity_index'
]

# 9 scalar/raster features
SCALAR_FEATURE_COLUMNS = [
    'soil_type', 'elevation_m', 'pop_density_persqkm', 'land_cover_class',
    'ndvi', 'annual_precip_mm', 'annual_mean_temp_c', 'mean_wind_speed_ms',
    'impervious_surface_pct'
]

# Statistics computed per array feature for the tree ensemble
STAT_NAMES = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skew']


class GeoVisionFusionModel:
    """
    Multi-model fusion prediction engine for GeoVision.
    Loads 4 model groups and runs the full inference pipeline:
      LSTM MIMO → Tree Ensemble → (CNN optional) → Fusion Meta-Learner
    """

    def __init__(self):
        """Initialize GeoVision fusion model."""
        model_root = os.getenv('MODEL_ROOT_PATH', '').strip()
        if model_root:
            self.MODEL_BASE_DIR = os.path.join(os.path.abspath(model_root), 'geovision')
        else:
            self.MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geovision')

        # Model directories
        self.LSTM_DIR     = os.path.join(self.MODEL_BASE_DIR, "lstm")
        self.ENSEMBLE_DIR = os.path.join(self.MODEL_BASE_DIR, "ensemble")
        self.CNN_DIR      = os.path.join(self.MODEL_BASE_DIR, "cnn")
        self.FUSION_DIR   = os.path.join(self.MODEL_BASE_DIR, "fusion")

        # Model components (loaded on demand)
        self.lstm_model = None
        self.lstm_embedding_model = None
        self.lstm_input_scaler = None

        self.ensemble_pipeline = None

        self.cnn_model = None

        self.fusion_disaster_model = None
        self.fusion_disaster_scaler = None
        self.fusion_weather_model = None
        self.fusion_weather_scaler = None
        self.fusion_feature_map = None

        self.models_loaded = False
        self._load_status = {}

        logger.info("[GEOVISION] Fusion model initialized")

    # ────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ────────────────────────────────────────────────────────────────
    def load_models(self) -> bool:
        """Load all upstream models and fusion meta-learners."""
        logger.info("[GEOVISION] ===========================================")
        logger.info("[GEOVISION]   LOADING FUSION PIPELINE MODELS")
        logger.info("[GEOVISION] ===========================================")

        success_count = 0
        total_models = 4

        # ── 1. LSTM MIMO ──
        success_count += self._load_lstm()
        # ── 2. Tree Ensemble ──
        success_count += self._load_ensemble()
        # ── 3. CNN ResNet50 (optional) ──
        success_count += self._load_cnn()
        # ── 4. Fusion Meta-Learners ──
        success_count += self._load_fusion()

        self.models_loaded = success_count >= 2  # Need at least LSTM + fusion
        logger.info(f"[GEOVISION] Models loaded: {success_count}/{total_models}")
        logger.info(f"[GEOVISION] Pipeline ready: {self.models_loaded}")
        return self.models_loaded

    def _load_lstm(self) -> int:
        """Load LSTM MIMO model + embedding extractor + scaler."""
        logger.info("[GEOVISION]   [1/4] Loading LSTM MIMO Model...")
        model_path = os.path.join(self.LSTM_DIR, 'temporal_lstm_mimo_model.keras')

        if not os.path.exists(model_path):
            logger.warning(f"[GEOVISION]   NOT FOUND: {model_path}")
            self._load_status['lstm'] = 'not_found'
            return 0

        try:
            import tensorflow as tf

            # Define TemporalAttentionPooling inline (avoids import dependency)
            class TemporalAttentionPooling(tf.keras.layers.Layer):
                def __init__(self, units=64, **kwargs):
                    super().__init__(**kwargs)
                    self.units = units
                def build(self, input_shape):
                    feature_dim = input_shape[-1]
                    self.W = self.add_weight('attention_W', (feature_dim, self.units), initializer='glorot_uniform')
                    self.b = self.add_weight('attention_b', (self.units,), initializer='zeros')
                    self.u = self.add_weight('attention_u', (self.units, 1), initializer='glorot_uniform')
                def call(self, inputs):
                    score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
                    attn = tf.nn.softmax(tf.tensordot(score, self.u, axes=[[2], [0]]), axis=1)
                    return tf.reduce_sum(inputs * attn, axis=1)
                def get_config(self):
                    config = super().get_config()
                    config.update({'units': self.units})
                    return config

            custom_objects = {'TemporalAttentionPooling': TemporalAttentionPooling}
            self.lstm_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

            # Create embedding extractor from the temporal_embedding layer
            self.lstm_embedding_model = tf.keras.Model(
                inputs=self.lstm_model.input,
                outputs=self.lstm_model.get_layer('temporal_embedding').output
            )
            logger.info(f"[GEOVISION]     [OK] LSTM model loaded")

        except Exception as e:
            logger.error(f"[GEOVISION]     [FAIL] LSTM model FAILED: {e}")
            self._load_status['lstm'] = f'error: {e}'
            return 0

        # Load LSTM input scaler
        scaler_path = os.path.join(self.LSTM_DIR, 'input_scaler.joblib')
        if os.path.exists(scaler_path):
            self.lstm_input_scaler = joblib.load(scaler_path)
            logger.info(f"[GEOVISION]     [OK] LSTM input scaler loaded")

        self._load_status['lstm'] = 'loaded'
        return 1

    def _load_ensemble(self) -> int:
        """Load tree ensemble pipeline (5 models + scaler + selector)."""
        logger.info("[GEOVISION]   [2/4] Loading Tree Ensemble Pipeline...")
        pipeline_path = os.path.join(self.ENSEMBLE_DIR, 'ensemble_5class_pipeline.joblib')

        if not os.path.exists(pipeline_path):
            logger.warning(f"[GEOVISION]   NOT FOUND: {pipeline_path}")
            self._load_status['ensemble'] = 'not_found'
            return 0

        try:
            self.ensemble_pipeline = joblib.load(pipeline_path)
            models = self.ensemble_pipeline.get('models', {})
            logger.info(f"[GEOVISION]     [OK] Ensemble loaded: {list(models.keys())}")
            self._load_status['ensemble'] = 'loaded'
            return 1
        except Exception as e:
            logger.error(f"[GEOVISION]     [FAIL] Ensemble FAILED: {e}")
            self._load_status['ensemble'] = f'error: {e}'
            return 0

    def _load_cnn(self) -> int:
        """Load CNN ResNet50 model (optional — used only with satellite imagery)."""
        logger.info("[GEOVISION]   [3/4] Loading CNN ResNet50 (optional)...")

        for name in ['best_resnet_finetuned.keras', 'best_resnet_model.keras', 'final_model.keras']:
            cnn_path = os.path.join(self.CNN_DIR, name)
            if os.path.exists(cnn_path):
                try:
                    import tensorflow as tf
                    self.cnn_model = tf.keras.models.load_model(cnn_path)
                    logger.info(f"[GEOVISION]     [OK] CNN loaded: {name}")
                    logger.info(f"[GEOVISION]     CNN input shape: {self.cnn_model.input_shape}")
                    self._load_status['cnn'] = 'loaded'
                    return 1
                except Exception as e:
                    logger.warning(f"[GEOVISION]     CNN {name} failed: {e}")

        logger.info("[GEOVISION]     CNN not loaded (satellite imagery needed at inference time)")
        self._load_status['cnn'] = 'skipped'
        return 0

    def _load_fusion(self) -> int:
        """Load fusion meta-learner models (disaster + weather heads)."""
        logger.info("[GEOVISION]   [4/4] Loading Fusion Meta-Learners...")
        loaded = 0

        # Disaster fusion
        dis_path = os.path.join(self.FUSION_DIR, 'fusion_disaster_model.pkl')
        if os.path.exists(dis_path):
            try:
                pkg = joblib.load(dis_path)
                self.fusion_disaster_model = pkg['model']
                self.fusion_disaster_scaler = pkg.get('scaler', None)
                logger.info(f"[GEOVISION]     [OK] Disaster fusion ({pkg.get('model_name', '?')})")
                loaded += 1
            except Exception as e:
                logger.error(f"[GEOVISION]     [FAIL] Disaster fusion FAILED: {e}")

        # Weather fusion
        wx_path = os.path.join(self.FUSION_DIR, 'fusion_weather_model.pkl')
        if os.path.exists(wx_path):
            try:
                pkg = joblib.load(wx_path)
                self.fusion_weather_model = pkg['model']
                self.fusion_weather_scaler = pkg.get('scaler', None)
                logger.info(f"[GEOVISION]     [OK] Weather fusion ({pkg.get('model_name', '?')})")
                loaded += 1
            except Exception as e:
                logger.error(f"[GEOVISION]     [FAIL] Weather fusion FAILED: {e}")

        # Feature map
        fmap_path = os.path.join(self.FUSION_DIR, 'fusion_feature_map.json')
        if os.path.exists(fmap_path):
            try:
                with open(fmap_path, 'r', encoding='utf-8') as f:
                    fmap_text = f.read()
                if fmap_text.lstrip().startswith('version https://git-lfs.github.com/spec/v1'):
                    logger.warning("[GEOVISION]     fusion_feature_map.json is a Git LFS pointer; continuing without feature map")
                    self.fusion_feature_map = None
                else:
                    self.fusion_feature_map = json.loads(fmap_text)
            except Exception as e:
                logger.warning(f"[GEOVISION]     fusion_feature_map.json invalid: {e}; continuing without feature map")
                self.fusion_feature_map = None

        self._load_status['fusion'] = 'loaded' if loaded == 2 else f'partial ({loaded}/2)'
        return 1 if loaded >= 1 else 0

    # ────────────────────────────────────────────────────────────────
    # LSTM PROCESSING
    # ────────────────────────────────────────────────────────────────
    def _process_lstm(self, weather_data: Dict[str, Any],
                      feature_data: Dict[str, Any]) -> Tuple[
                          Optional[np.ndarray], Optional[np.ndarray],
                          Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process weather time-series through LSTM MIMO.
        
        Returns (disaster_probs, weather_probs, embeddings, forecast) or all None.
        """
        if self.lstm_model is None:
            logger.warning("[GEOVISION] LSTM not loaded — skipping")
            return None, None, None, None

        logger.info("[GEOVISION]   STEP 1: LSTM MIMO processing...")

        # Build temporal sequence: (1, 60, 36)
        x_sample = []
        matched_features = []
        missing_features = []
        for feature_name in INPUT_FEATURES:
            seq = None
            # Try weather_data first, then feature_data
            for source in [weather_data, feature_data]:
                if feature_name in source and isinstance(source[feature_name], list):
                    raw = source[feature_name]
                    arr = np.array(raw, dtype=float)
                    # Pad or truncate to SEQUENCE_LENGTH
                    if len(arr) < SEQUENCE_LENGTH:
                        padded = np.full(SEQUENCE_LENGTH, np.nan)
                        padded[:len(arr)] = arr
                        seq = padded
                    else:
                        seq = arr[:SEQUENCE_LENGTH]
                    break
            if seq is None:
                seq = np.full(SEQUENCE_LENGTH, np.nan)
                missing_features.append(feature_name)
            else:
                matched_features.append(feature_name)
            x_sample.append(seq)

        logger.info(f"[GEOVISION]     LSTM features matched: {len(matched_features)}/{len(INPUT_FEATURES)}")
        if missing_features:
            logger.warning(f"[GEOVISION]     LSTM missing features ({len(missing_features)}): {missing_features[:10]}")

        X = np.array(x_sample).T  # (60, 36)
        X = X.reshape(1, SEQUENCE_LENGTH, len(INPUT_FEATURES))  # (1, 60, 36)

        # Normalize with saved scaler
        if self.lstm_input_scaler is not None:
            X_means = np.nanmean(X, axis=(0, 1))
            X_filled = np.where(np.isnan(X), X_means, X)
            X_reshaped = X_filled.reshape(-1, X_filled.shape[-1])
            X_scaled = self.lstm_input_scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            X_scaled = np.where(np.isnan(X), MASK_VALUE, X_scaled)
        else:
            X_scaled = np.nan_to_num(X, nan=MASK_VALUE)

        # Predict
        disaster_probs, weather_probs, forecast = self.lstm_model.predict(X_scaled, verbose=0)
        embeddings = self.lstm_embedding_model.predict(X_scaled, verbose=0)

        logger.info(f"[GEOVISION]     LSTM disaster probs: {disaster_probs.shape}")
        logger.info(f"[GEOVISION]     LSTM embeddings: {embeddings.shape}")

        return disaster_probs, weather_probs, embeddings, forecast

    # ────────────────────────────────────────────────────────────────
    # TREE ENSEMBLE PROCESSING
    # ────────────────────────────────────────────────────────────────
    def _compute_stats(self, arr: np.ndarray) -> List[float]:
        """Compute 8 summary statistics matching the ensemble training script."""
        arr = np.array(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return [0.0] * 8
        # Compute skewness manually to avoid scipy dependency
        m = float(np.mean(arr))
        s = float(np.std(arr))
        skewness = float(np.mean(((arr - m) / s) ** 3)) if s > 0 else 0.0
        return [
            m,
            s,
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
            skewness
        ]

    def _process_ensemble(self, weather_data: Dict[str, Any],
                          feature_data: Dict[str, Any],
                          raster_data: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Process tabular features through tree ensemble.
        Returns ensemble_probs (1, 5) or None.
        """
        if self.ensemble_pipeline is None:
            logger.warning("[GEOVISION] Ensemble not loaded — skipping")
            return None

        logger.info("[GEOVISION]   STEP 2: Tree Ensemble processing...")

        models = self.ensemble_pipeline.get('models', {})
        scaler = self.ensemble_pipeline.get('scaler', None)
        selector = self.ensemble_pipeline.get('selector', None)
        imputation_values = self.ensemble_pipeline.get('imputation_values', {})
        horizon = self.ensemble_pipeline.get('horizon', HORIZON)

        # Build tabular features: 36 arrays × 8 stats + 9 scalar = ~297
        row_data = {}
        for col in INPUT_FEATURES:
            raw = None
            for source in [weather_data, feature_data]:
                if col in source and isinstance(source[col], list):
                    raw = np.array(source[col], dtype=float)
                    break
            if raw is not None:
                arr_trimmed = raw[:SEQUENCE_LENGTH - horizon]  # First 59 days
                stats = self._compute_stats(arr_trimmed)
            else:
                stats = [0.0] * 8
            for s_name, s_val in zip(STAT_NAMES, stats):
                row_data[f'{col}_{s_name}'] = s_val

        # Add scalar/raster features
        for col in SCALAR_FEATURE_COLUMNS:
            val = raster_data.get(col, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            if val == -9999 or val == -9999.0:
                val = 0.0
            row_data[col] = float(val)

        X_df = pd.DataFrame([row_data])

        # Impute
        for col, imp_val in imputation_values.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(imp_val)
        X_df = X_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Feature selection — reorder columns to match selector's training order
        if selector is not None:
            try:
                # Reorder X_df columns to match the selector's expected feature order
                expected_cols = list(selector.feature_names_in_)
                # Add any missing columns with 0.0
                for c in expected_cols:
                    if c not in X_df.columns:
                        X_df[c] = 0.0
                X_df = X_df[expected_cols]
                X_selected = selector.transform(X_df)
                logger.info(f"[GEOVISION]     Feature selection OK: {X_df.shape[1]} -> {X_selected.shape[1]}")
            except Exception as e:
                logger.warning(f"[GEOVISION]     Feature selection error: {e}, using raw")
                X_selected = X_df.values
        else:
            X_selected = X_df.values

        # Scale
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_selected)
            except Exception:
                X_scaled = X_selected
        else:
            X_scaled = X_selected

        # Average probabilities across ensemble models
        all_probas = []
        for model_name, model in models.items():
            try:
                proba = model.predict_proba(X_scaled)
                all_probas.append(proba)
            except Exception as e:
                logger.warning(f"[GEOVISION]     {model_name} failed: {e}")

        if not all_probas:
            return None

        ensemble_probs = np.mean(all_probas, axis=0)
        logger.info(f"[GEOVISION]     Ensemble probs: {ensemble_probs.shape} ({len(all_probas)} models)")
        return ensemble_probs

    # ────────────────────────────────────────────────────────────────
    # CNN PROCESSING
    # ────────────────────────────────────────────────────────────────
    def _process_cnn(self, satellite_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a preprocessed 6-band satellite image through CNN ResNet50.

        Args:
            satellite_image: numpy array of shape (1, 224, 224, 6), normalized to [0,1]

        Returns:
            cnn_probs: (1, 5) disaster class probabilities, or None on failure
        """
        if self.cnn_model is None:
            logger.warning("[GEOVISION] CNN model not loaded — skipping")
            return None

        logger.info("[GEOVISION]   STEP 2b: CNN ResNet50 processing...")

        try:
            cnn_probs = self.cnn_model.predict(satellite_image, verbose=0)
            logger.info(f"[GEOVISION]     CNN probs shape: {cnn_probs.shape}")
            logger.info(f"[GEOVISION]     CNN top class: {DISASTER_CLASSES[int(np.argmax(cnn_probs[0]))]}")
            return cnn_probs
        except Exception as e:
            logger.error(f"[GEOVISION]     CNN inference FAILED: {e}")
            return None

    # ────────────────────────────────────────────────────────────────
    # FUSION STACKING + META-LEARNER
    # ────────────────────────────────────────────────────────────────
    def _stack_and_predict(self,
                           lstm_dis_probs: np.ndarray,
                           lstm_wx_probs: np.ndarray,
                           lstm_embeddings: np.ndarray,
                           ensemble_probs: Optional[np.ndarray] = None,
                           cnn_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Stack intermediate outputs and run fusion meta-learner.
        
        Returns dict with final predictions.
        """
        logger.info("[GEOVISION]   STEP 3: Fusion Meta-Learner...")

        # Stack features: LSTM disaster (5) + LSTM weather (5) + embeddings (128) + [ensemble (5)] + [cnn (5)]
        parts = [lstm_dis_probs, lstm_wx_probs, lstm_embeddings]
        if cnn_probs is not None and cnn_probs.shape[0] == lstm_dis_probs.shape[0]:
            parts.append(cnn_probs)
        if ensemble_probs is not None and ensemble_probs.shape[0] == lstm_dis_probs.shape[0]:
            parts.append(ensemble_probs)

        fusion_features = np.concatenate(parts, axis=1)
        logger.info(f"[GEOVISION]     Fusion feature vector: {fusion_features.shape}")

        result = {
            'fusion_features_dim': fusion_features.shape[1],
        }

        # ── Disaster prediction ──
        if self.fusion_disaster_model is not None:
            try:
                fusion_scaled = self._scale_fusion(fusion_features, self.fusion_disaster_scaler)
                dis_pred = self.fusion_disaster_model.predict(fusion_scaled)
                dis_proba = self.fusion_disaster_model.predict_proba(fusion_scaled)
                result['disaster_prediction'] = DISASTER_CLASSES[int(dis_pred[0])] if int(dis_pred[0]) < len(DISASTER_CLASSES) else f'Class_{dis_pred[0]}'
                result['disaster_probabilities'] = {
                    cls: round(float(dis_proba[0, i]), 4) for i, cls in enumerate(DISASTER_CLASSES)
                    if i < dis_proba.shape[1]
                }
                result['disaster_confidence'] = round(float(np.max(dis_proba[0])), 4)
                logger.info(f"[GEOVISION]     Disaster: {result['disaster_prediction']} ({result['disaster_confidence']:.2%})")
            except Exception as e:
                logger.error(f"[GEOVISION]     Disaster fusion FAILED: {e}")
                # Fallback to LSTM
                idx = int(np.argmax(lstm_dis_probs[0]))
                result['disaster_prediction'] = DISASTER_CLASSES[idx]
                result['disaster_probabilities'] = {
                    cls: round(float(lstm_dis_probs[0, i]), 4) for i, cls in enumerate(DISASTER_CLASSES)
                    if i < lstm_dis_probs.shape[1]
                }
                result['disaster_confidence'] = round(float(np.max(lstm_dis_probs[0])), 4)
                result['disaster_source'] = 'lstm_fallback'
        else:
            idx = int(np.argmax(lstm_dis_probs[0]))
            result['disaster_prediction'] = DISASTER_CLASSES[idx]
            result['disaster_probabilities'] = {
                cls: round(float(lstm_dis_probs[0, i]), 4) for i, cls in enumerate(DISASTER_CLASSES)
                if i < lstm_dis_probs.shape[1]
            }
            result['disaster_confidence'] = round(float(np.max(lstm_dis_probs[0])), 4)
            result['disaster_source'] = 'lstm_only'

        # ── Weather prediction ──
        if self.fusion_weather_model is not None:
            try:
                fusion_wx_scaled = self._scale_fusion(fusion_features, self.fusion_weather_scaler)
                wx_pred = self.fusion_weather_model.predict(fusion_wx_scaled)
                wx_proba = self.fusion_weather_model.predict_proba(fusion_wx_scaled)
                result['weather_prediction'] = WEATHER_REGIMES[int(wx_pred[0])] if int(wx_pred[0]) < len(WEATHER_REGIMES) else f'Regime_{wx_pred[0]}'
                result['weather_probabilities'] = {
                    cls: round(float(wx_proba[0, i]), 4) for i, cls in enumerate(WEATHER_REGIMES)
                    if i < wx_proba.shape[1]
                }
                result['weather_confidence'] = round(float(np.max(wx_proba[0])), 4)
                logger.info(f"[GEOVISION]     Weather: {result['weather_prediction']} ({result['weather_confidence']:.2%})")
            except Exception as e:
                logger.error(f"[GEOVISION]     Weather fusion FAILED: {e}")
                idx = int(np.argmax(lstm_wx_probs[0]))
                result['weather_prediction'] = WEATHER_REGIMES[idx]
                result['weather_probabilities'] = {
                    cls: round(float(lstm_wx_probs[0, i]), 4) for i, cls in enumerate(WEATHER_REGIMES)
                    if i < lstm_wx_probs.shape[1]
                }
                result['weather_confidence'] = round(float(np.max(lstm_wx_probs[0])), 4)
                result['weather_source'] = 'lstm_fallback'
        else:
            idx = int(np.argmax(lstm_wx_probs[0]))
            result['weather_prediction'] = WEATHER_REGIMES[idx]
            result['weather_probabilities'] = {
                cls: round(float(lstm_wx_probs[0, i]), 4) for i, cls in enumerate(WEATHER_REGIMES)
                if i < lstm_wx_probs.shape[1]
            }
            result['weather_confidence'] = round(float(np.max(lstm_wx_probs[0])), 4)
            result['weather_source'] = 'lstm_only'

        return result

    def _scale_fusion(self, features: np.ndarray, scaler) -> np.ndarray:
        """Scale fusion features, handling dimension mismatches via padding/trimming."""
        if scaler is None:
            return features
        try:
            return scaler.transform(features)
        except Exception:
            expected_dim = scaler.n_features_in_
            if features.shape[1] < expected_dim:
                padding = np.zeros((features.shape[0], expected_dim - features.shape[1]))
                return scaler.transform(np.concatenate([features, padding], axis=1))
            elif features.shape[1] > expected_dim:
                return scaler.transform(features[:, :expected_dim])
            return features

    # ────────────────────────────────────────────────────────────────
    # MAIN PREDICTION METHOD
    # ────────────────────────────────────────────────────────────────
    def predict(self, weather_data: Dict[str, Any],
                feature_data: Dict[str, Any],
                raster_data: Dict[str, float],
                satellite_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run the full GeoVision fusion inference pipeline.

        Args:
            weather_data: 17 raw weather arrays (each 60 values)
            feature_data: 19 engineered feature arrays (each 60 values)
            raster_data:  9 scalar raster features
            satellite_image: Optional preprocessed (1, 224, 224, 6) array for CNN

        Returns:
            Comprehensive prediction result dict.
        """
        if not self.models_loaded:
            return {
                'success': False,
                'error': 'Models not loaded. Call load_models() first.'
            }

        logger.info("[GEOVISION] ===========================================")
        logger.info("[GEOVISION]   RUNNING FUSION INFERENCE PIPELINE")
        logger.info("[GEOVISION] ===========================================")

        try:
            # Step 1: LSTM
            lstm_dis, lstm_wx, lstm_emb, lstm_forecast = self._process_lstm(weather_data, feature_data)
            if lstm_dis is None:
                return {
                    'success': False,
                    'error': 'LSTM processing failed — required for fusion.'
                }

            # Step 2: Tree Ensemble
            ensemble_probs = self._process_ensemble(weather_data, feature_data, raster_data)

            # Step 3: CNN ResNet50 (if satellite imagery is available)
            cnn_probs = None
            if satellite_image is not None:
                cnn_probs = self._process_cnn(satellite_image)

            # Step 4: Fusion Meta-Learner
            fusion_result = self._stack_and_predict(
                lstm_dis, lstm_wx, lstm_emb,
                ensemble_probs, cnn_probs
            )

            # Build intermediate results
            models_used = ['LSTM_MIMO']
            if ensemble_probs is not None:
                models_used.append('Tree_Ensemble')
            if cnn_probs is not None:
                models_used.append('CNN_ResNet50')
            models_used.append('Fusion_MetaLearner')

            intermediate = {
                'lstm_disaster_probs': {
                    cls: round(float(lstm_dis[0, i]), 4) for i, cls in enumerate(DISASTER_CLASSES)
                    if i < lstm_dis.shape[1]
                },
                'lstm_weather_probs': {
                    cls: round(float(lstm_wx[0, i]), 4) for i, cls in enumerate(WEATHER_REGIMES)
                    if i < lstm_wx.shape[1]
                },
                'models_used': models_used,
            }
            if ensemble_probs is not None:
                intermediate['ensemble_disaster_probs'] = {
                    cls: round(float(ensemble_probs[0, i]), 4) for i, cls in enumerate(DISASTER_CLASSES)
                    if i < ensemble_probs.shape[1]
                }

            return {
                'success': True,
                'disaster_prediction': fusion_result['disaster_prediction'],
                'disaster_probabilities': fusion_result['disaster_probabilities'],
                'disaster_confidence': fusion_result['disaster_confidence'],
                'weather_prediction': fusion_result['weather_prediction'],
                'weather_probabilities': fusion_result['weather_probabilities'],
                'weather_confidence': fusion_result['weather_confidence'],
                'intermediate': intermediate,
                'metadata': {
                    'fusion_features_dim': fusion_result.get('fusion_features_dim'),
                    'models_loaded': self._load_status,
                    'disaster_source': fusion_result.get('disaster_source', 'fusion'),
                    'weather_source': fusion_result.get('weather_source', 'fusion'),
                }
            }

        except Exception as e:
            logger.error(f"[GEOVISION] Pipeline error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Fusion pipeline error: {str(e)}'
            }

    def get_model_status(self) -> Dict[str, Any]:
        """Return status of all loaded models."""
        return {
            'models_loaded': self.models_loaded,
            'components': self._load_status,
            'disaster_classes': DISASTER_CLASSES,
            'weather_regimes': WEATHER_REGIMES,
            'features': {
                'temporal': len(INPUT_FEATURES),
                'scalar': len(SCALAR_FEATURE_COLUMNS),
                'total_tabular': len(INPUT_FEATURES) * len(STAT_NAMES) + len(SCALAR_FEATURE_COLUMNS),
                'sequence_length': SEQUENCE_LENGTH,
                'embedding_dim': EMBEDDING_DIM,
            }
        }
