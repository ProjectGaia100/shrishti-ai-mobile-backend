"""
Raster Data Configuration for HazardGuard System
Configuration loader for raster data paths and settings

Updated to use COG-optimized raster files from final_lookup_tables/
All 9 raster datasets are Cloud Optimized GeoTIFF (ZSTD compressed, 256x256 tiles)
"""

import os
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Base directory of the backend (where main.py lives)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Google Cloud Storage bucket (public, COG-optimised) ----------
GCS_BUCKET_BASE_URL = os.getenv(
    'GCS_BUCKET_BASE_URL',
    'https://storage.googleapis.com/satellite-cog-data-for-shrishti'
)

# Local fallback (kept for offline / dev use)
LOCAL_LOOKUP_DIR = os.path.join(os.path.dirname(BACKEND_DIR), 'final_lookup_tables')

# Mapping from config key -> COG filename
DEFAULT_COG_FILES = {
    'soil': 'soil_type.tif',
    'elevation': 'elevation.tif',
    'population': 'population_density.tif',
    'landcover': 'land_cover.tif',
    'ndvi': 'ndvi.tif',
    'precip': 'annual_precip.tif',
    'temp': 'mean_annual_temp.tif',
    'wind': 'wind_speed.tif',
    'impervious': 'impervious_surface.tif',
}


def _is_url(path: str) -> bool:
    """Check if a path is an HTTP(S) URL."""
    return path.startswith('http://') or path.startswith('https://')


def _path_exists(path: str) -> bool:
    """Check existence — works for both local paths and URLs.
    For URLs we do a lightweight HEAD request (COG files are public).
    """
    if _is_url(path):
        try:
            import requests
            resp = requests.head(path, timeout=5, allow_redirects=True)
            return resp.status_code == 200
        except Exception:
            return False
    return os.path.exists(path)


def _resolve_raster_path(env_value: str) -> str:
    """Resolve a raster path from an env value.
    
    URLs (http/https) are returned as-is.
    If the value is an absolute path, return as-is.
    If relative, resolve relative to BACKEND_DIR.
    """
    if not env_value:
        return ''
    env_value = env_value.strip()
    # URLs must not be touched by os.path helpers
    if _is_url(env_value):
        return env_value
    if os.path.isabs(env_value):
        return os.path.normpath(env_value)
    return os.path.normpath(os.path.join(BACKEND_DIR, env_value))


class RasterDataConfig:
    """Configuration manager for raster data sources"""
    
    def __init__(self, env_path: Optional[str] = None):
        """Initialize raster data configuration"""
        self.env_path = env_path or '.env'
        self.config = {}
        
        # Load environment variables
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from environment variables"""
        try:
            # Load .env file if it exists
            if os.path.exists(self.env_path):
                load_dotenv(self.env_path)
                logger.info(f"Loaded environment variables from {self.env_path}")
            else:
                logger.warning(f"Environment file not found: {self.env_path}")
            
            # Build default raster paths — prefer GCS bucket, fall back to local
            default_paths = {}
            for key, filename in DEFAULT_COG_FILES.items():
                gcs_url = f"{GCS_BUCKET_BASE_URL}/{filename}"
                local_path = os.path.join(LOCAL_LOOKUP_DIR, filename)
                if os.path.exists(local_path):
                    # Keep local as a fast fallback (no network latency)
                    # but GCS is the primary source for deployment
                    default_paths[key] = gcs_url
                    logger.debug(f"{key}: Using GCS URL {gcs_url} (local copy exists)")
                else:
                    default_paths[key] = gcs_url
                    logger.debug(f"{key}: Using GCS URL {gcs_url}")
            
            # Environment variables override defaults; resolve relative paths
            env_key_map = {
                'soil': 'RASTER_SOIL_PATH',
                'elevation': 'RASTER_ELEVATION_PATH',
                'population': 'RASTER_POPULATION_PATH',
                'landcover': 'RASTER_LANDCOVER_PATH',
                'ndvi': 'RASTER_NDVI_PATH',
                'precip': 'RASTER_PRECIP_PATH',
                'temp': 'RASTER_TEMP_PATH',
                'wind': 'RASTER_WIND_PATH',
                'impervious': 'RASTER_IMPERVIOUS_PATH',
            }
            
            raster_paths = {}
            for key, env_var in env_key_map.items():
                env_val = os.getenv(env_var, '')
                if env_val:
                    resolved = _resolve_raster_path(env_val)
                    raster_paths[key] = resolved
                else:
                    raster_paths[key] = default_paths.get(key, '')
            
            self.config = {
                'raster_paths': raster_paths,
                'hwsd2_smu_path': _resolve_raster_path(os.getenv('HWSD2_SMU_PATH', '')),
                'hwsd2_wrb4_path': _resolve_raster_path(os.getenv('HWSD2_WRB4_PATH', '')),
                'batch_size': int(os.getenv('RASTER_BATCH_SIZE', '100')),
                'enable_logging': os.getenv('RASTER_ENABLE_LOGGING', 'true').lower() == 'true',
                'log_level': os.getenv('RASTER_LOG_LEVEL', 'INFO').upper(),
                'cache_enabled': os.getenv('RASTER_CACHE_ENABLED', 'false').lower() == 'true',
                'cache_timeout': int(os.getenv('RASTER_CACHE_TIMEOUT', '3600'))
            }
            
            # Filter out empty paths
            self.config['raster_paths'] = {
                k: v for k, v in self.config['raster_paths'].items() 
                if v and v.strip()
            }
            
            gcs_count = sum(1 for v in self.config['raster_paths'].values() if _is_url(v))
            local_count = len(self.config['raster_paths']) - gcs_count
            logger.info(f"Loaded configuration for {len(self.config['raster_paths'])} raster sources "
                        f"({gcs_count} GCS, {local_count} local)")
            
        except Exception as e:
            logger.error(f"Error loading raster configuration: {e}")
            self.config = self.get_default_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary"""
        return self.config.copy()
    
    def get_raster_paths(self) -> Dict[str, str]:
        """Get raster file paths"""
        return self.config.get('raster_paths', {})
    
    def get_soil_database_paths(self) -> Dict[str, str]:
        """Get soil database file paths"""
        return {
            'hwsd2_smu_path': self.config.get('hwsd2_smu_path', ''),
            'hwsd2_wrb4_path': self.config.get('hwsd2_wrb4_path', '')
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and file paths"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'raster_files': {},
            'database_files': {}
        }
        
        try:
            # Validate raster files (supports both local paths and HTTPS URLs)
            for data_type, file_path in self.config.get('raster_paths', {}).items():
                file_status = {
                    'path': file_path,
                    'exists': False,
                    'readable': False,
                    'size_mb': 0,
                    'is_remote': _is_url(file_path),
                    'error': None
                }
                
                if not file_path:
                    file_status['error'] = 'Path not configured'
                    validation_results['warnings'].append(f"{data_type}: Path not configured")
                elif not _path_exists(file_path):
                    file_status['error'] = 'File does not exist'
                    validation_results['errors'].append(f"{data_type}: File does not exist - {file_path}")
                    validation_results['valid'] = False
                else:
                    file_status['exists'] = True
                    try:
                        if not _is_url(file_path):
                            file_status['size_mb'] = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        
                        # Try to open with rasterio (works with both local + HTTPS COGs)
                        import rasterio
                        with rasterio.open(file_path) as src:
                            file_status['readable'] = True
                            file_status['crs'] = str(src.crs)
                            file_status['shape'] = src.shape
                    except ImportError:
                        validation_results['warnings'].append("rasterio not installed - cannot validate raster files")
                    except Exception as e:
                        file_status['error'] = f"Cannot read file: {str(e)}"
                        validation_results['errors'].append(f"{data_type}: Cannot read file - {str(e)}")
                
                validation_results['raster_files'][data_type] = file_status
            
            # Validate soil database files
            for db_name, file_path in self.get_soil_database_paths().items():
                file_status = {
                    'path': file_path,
                    'exists': False,
                    'readable': False,
                    'size_mb': 0,
                    'error': None
                }
                
                if not file_path:
                    file_status['error'] = 'Path not configured'
                    validation_results['warnings'].append(f"{db_name}: Path not configured")
                elif not os.path.exists(file_path):
                    file_status['error'] = 'File does not exist'
                    validation_results['errors'].append(f"{db_name}: File does not exist - {file_path}")
                    validation_results['valid'] = False
                else:
                    file_status['exists'] = True
                    try:
                        file_status['size_mb'] = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        
                        # Try to open with pandas
                        import pandas as pd
                        if file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                            file_status['readable'] = True
                            file_status['rows'] = len(df)
                            file_status['columns'] = list(df.columns)
                        else:
                            file_status['error'] = 'Unsupported file format'
                    except ImportError:
                        validation_results['warnings'].append("pandas not installed - cannot validate Excel files")
                    except Exception as e:
                        file_status['error'] = f"Cannot read file: {str(e)}"
                        validation_results['errors'].append(f"{db_name}: Cannot read file - {str(e)}")
                
                validation_results['database_files'][db_name] = file_status
            
            # Summary
            validation_results['summary'] = {
                'total_raster_files': len(self.config.get('raster_paths', {})),
                'available_raster_files': sum(1 for f in validation_results['raster_files'].values() if f['exists']),
                'readable_raster_files': sum(1 for f in validation_results['raster_files'].values() if f['readable']),
                'total_database_files': len(self.get_soil_database_paths()),
                'available_database_files': sum(1 for f in validation_results['database_files'].values() if f['exists']),
                'readable_database_files': sum(1 for f in validation_results['database_files'].values() if f['readable'])
            }
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Error validating raster configuration: {e}")
        
        return validation_results
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when loading fails"""
        return {
            'raster_paths': {},
            'hwsd2_smu_path': '',
            'hwsd2_wrb4_path': '',
            'batch_size': 100,
            'enable_logging': True,
            'log_level': 'INFO',
            'cache_enabled': False,
            'cache_timeout': 3600
        }
    
    def get_feature_availability(self) -> Dict[str, bool]:
        """Get availability status for each feature"""
        feature_mapping = {
            'soil_type': 'soil',
            'elevation_m': 'elevation',
            'pop_density_persqkm': 'population',
            'land_cover_class': 'landcover',
            'ndvi': 'ndvi',
            'annual_precip_mm': 'precip',
            'annual_mean_temp_c': 'temp',
            'mean_wind_speed_ms': 'wind',
            'impervious_surface_pct': 'impervious'
        }
        
        raster_paths = self.get_raster_paths()
        availability = {}
        
        for feature_name, path_key in feature_mapping.items():
            path = raster_paths.get(path_key, '')
            # For remote URLs, assume available if configured
            availability[feature_name] = bool(path and (_is_url(path) or os.path.exists(path)))
        
        return availability
    
    def reload_config(self) -> bool:
        """Reload configuration from environment file"""
        try:
            self.load_config()
            logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration programmatically"""
        try:
            # Deep update
            for key, value in updates.items():
                if key in self.config:
                    if isinstance(self.config[key], dict) and isinstance(value, dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
                else:
                    self.config[key] = value
            
            logger.info("Configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for API responses"""
        return {
            'raster_sources_configured': len(self.config.get('raster_paths', {})),
            'soil_databases_configured': bool(
                self.config.get('hwsd2_smu_path') and 
                self.config.get('hwsd2_wrb4_path')
            ),
            'batch_size': self.config.get('batch_size', 100),
            'cache_enabled': self.config.get('cache_enabled', False),
            'logging_enabled': self.config.get('enable_logging', True),
            'available_features': list(self.get_feature_availability().keys())
        }

# Global configuration instance
raster_config = RasterDataConfig()

def get_raster_config() -> RasterDataConfig:
    """Get global raster configuration instance"""
    return raster_config

def reload_raster_config() -> bool:
    """Reload global raster configuration"""
    global raster_config
    return raster_config.reload_config()

def validate_raster_config() -> Dict[str, Any]:
    """Validate global raster configuration"""
    return raster_config.validate_configuration()