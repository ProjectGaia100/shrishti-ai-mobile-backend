"""
Raster Data Service for HazardGuard System
Business logic layer for raster data extraction and processing
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any, Union
from datetime import datetime
import os
import json

from models.raster_data_model import RasterDataModel

logger = logging.getLogger(__name__)


def _is_url(path: str) -> bool:
    return path.startswith('http://') or path.startswith('https://')

def _path_or_url_exists(path: str) -> bool:
    """Check existence for local path or public HTTPS URL."""
    if _is_url(path):
        return True  # assume public bucket URLs are available when configured
    return os.path.exists(path)


class RasterDataService:
    """Service layer for raster data operations"""
    
    def __init__(self, raster_config: Optional[Dict[str, str]] = None):
        """Initialize raster data service"""
        self.model = RasterDataModel()
        self.raster_config = raster_config or {}
        self.processing_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'last_extraction_time': None
        }
        
        # Initialize soil databases if paths are provided
        if self.raster_config.get('hwsd2_smu_path') and self.raster_config.get('hwsd2_wrb4_path'):
            self.initialize_soil_databases()
    
    def initialize_soil_databases(self) -> bool:
        """Initialize soil classification databases"""
        try:
            hwsd2_path = self.raster_config.get('hwsd2_smu_path')
            wrb4_path = self.raster_config.get('hwsd2_wrb4_path')
            
            if not hwsd2_path or not wrb4_path:
                logger.warning("Soil database paths not provided in configuration")
                return False
            
            success = self.model.load_soil_databases(hwsd2_path, wrb4_path)
            if success:
                logger.info("Soil databases initialized successfully")
            else:
                logger.error("Failed to initialize soil databases")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing soil databases: {e}")
            return False
    
    def validate_coordinates(self, coordinates: List[Dict[str, float]]) -> Tuple[bool, str]:
        """Validate coordinate input format"""
        try:
            if not coordinates or not isinstance(coordinates, list):
                return False, "Coordinates must be a non-empty list"
            
            coords_tuples = []
            for i, coord in enumerate(coordinates):
                if not isinstance(coord, dict):
                    return False, f"Coordinate {i} must be a dictionary with 'longitude' and 'latitude' keys"
                
                if 'longitude' not in coord or 'latitude' not in coord:
                    return False, f"Coordinate {i} missing required 'longitude' or 'latitude' key"
                
                try:
                    lon = float(coord['longitude'])
                    lat = float(coord['latitude'])
                    coords_tuples.append((lon, lat))
                except (ValueError, TypeError):
                    return False, f"Coordinate {i} longitude/latitude must be numeric"
            
            # Validate coordinate ranges
            if not self.model.validate_coordinates(coords_tuples):
                return False, "Coordinates contain invalid longitude (-180 to 180) or latitude (-90 to 90) values"
            
            return True, "Coordinates are valid"
            
        except Exception as e:
            return False, f"Error validating coordinates: {str(e)}"
    
    def process_raster_extraction(self, coordinates: List[Dict[str, float]], 
                                features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process raster data extraction for given coordinates"""
        start_time = datetime.now()
        
        try:
            self.processing_stats['total_extractions'] += 1
            
            # Validate coordinates
            is_valid, validation_message = self.validate_coordinates(coordinates)
            if not is_valid:
                self.processing_stats['failed_extractions'] += 1
                return {
                    'success': False,
                    'error': validation_message,
                    'data': None
                }
            
            # Convert to tuple format for model
            coords_tuples = [(float(coord['longitude']), float(coord['latitude'])) for coord in coordinates]
            
            # Check if raster paths are configured
            raster_paths = self._get_raster_paths()
            if not raster_paths:
                self.processing_stats['failed_extractions'] += 1
                return {
                    'success': False,
                    'error': "Raster data paths not configured",
                    'data': None
                }
            
            # Filter raster paths based on requested features
            if features:
                feature_path_mapping = {
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
                
                filtered_paths = {}
                for feature in features:
                    path_key = feature_path_mapping.get(feature)
                    if path_key and path_key in raster_paths:
                        filtered_paths[path_key] = raster_paths[path_key]
                
                raster_paths = filtered_paths
            
            # Extract raster features
            extracted_data = self.model.extract_all_features(coords_tuples, raster_paths)
            
            # Prepare response
            result_data = []
            for i, coord in enumerate(coordinates):
                coord_data = {
                    'longitude': coord['longitude'],
                    'latitude': coord['latitude']
                }
                
                # Add extracted features
                for feature_name, values in extracted_data.items():
                    if i < len(values):
                        coord_data[feature_name] = values[i]
                
                result_data.append(coord_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.processing_stats['successful_extractions'] += 1
            self.processing_stats['last_extraction_time'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'data': result_data,
                'metadata': {
                    'coordinates_processed': len(coordinates),
                    'features_extracted': len(extracted_data),
                    'processing_time_seconds': processing_time,
                    'extraction_timestamp': start_time.isoformat()
                }
            }
            
        except Exception as e:
            self.processing_stats['failed_extractions'] += 1
            logger.error(f"Error in raster extraction: {e}")
            return {
                'success': False,
                'error': f"Raster extraction failed: {str(e)}",
                'data': None
            }
    
    def process_batch_extraction(self, coordinates: List[Dict[str, float]], 
                               batch_size: int = 100, 
                               features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process large coordinate sets in batches"""
        start_time = datetime.now()
        
        try:
            all_results = []
            batch_count = 0
            
            for i in range(0, len(coordinates), batch_size):
                batch_coords = coordinates[i:i + batch_size]
                batch_result = self.process_raster_extraction(batch_coords, features)
                
                if batch_result['success']:
                    all_results.extend(batch_result['data'])
                    batch_count += 1
                else:
                    logger.error(f"Batch {batch_count + 1} failed: {batch_result['error']}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'data': all_results,
                'metadata': {
                    'total_coordinates': len(coordinates),
                    'batch_size': batch_size,
                    'batches_processed': batch_count,
                    'coordinates_processed': len(all_results),
                    'processing_time_seconds': processing_time,
                    'extraction_timestamp': start_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in batch raster extraction: {e}")
            return {
                'success': False,
                'error': f"Batch raster extraction failed: {str(e)}",
                'data': None
            }
    
    def create_raster_dataframe(self, coordinates: List[Dict[str, float]], 
                              features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create pandas DataFrame from raster extraction results"""
        try:
            result = self.process_raster_extraction(coordinates, features)
            
            if not result['success']:
                return result
            
            # Convert to DataFrame
            df = pd.DataFrame(result['data'])
            
            # Get DataFrame info
            buffer = []
            df.info(buf=buffer)
            df_info = '\n'.join(buffer)
            
            return {
                'success': True,
                'dataframe': df,
                'metadata': {
                    **result['metadata'],
                    'dataframe_shape': df.shape,
                    'dataframe_columns': df.columns.tolist(),
                    'dataframe_info': df_info
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating raster DataFrame: {e}")
            return {
                'success': False,
                'error': f"DataFrame creation failed: {str(e)}",
                'dataframe': None
            }
    
    def export_raster_data(self, coordinates: List[Dict[str, float]], 
                          format: str = 'json', 
                          features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export raster data in various formats"""
        try:
            result = self.process_raster_extraction(coordinates, features)
            
            if not result['success']:
                return result
            
            if format.lower() == 'json':
                exported_data = json.dumps(result['data'], indent=2)
                content_type = 'application/json'
                
            elif format.lower() == 'csv':
                df = pd.DataFrame(result['data'])
                exported_data = df.to_csv(index=False)
                content_type = 'text/csv'
                
            elif format.lower() == 'excel':
                df = pd.DataFrame(result['data'])
                # For Excel, we'd typically save to a file, but returning DataFrame for API
                exported_data = df
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                
            else:
                return {
                    'success': False,
                    'error': f"Unsupported export format: {format}. Supported: json, csv, excel",
                    'data': None
                }
            
            return {
                'success': True,
                'data': exported_data,
                'metadata': {
                    **result['metadata'],
                    'export_format': format,
                    'content_type': content_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error exporting raster data: {e}")
            return {
                'success': False,
                'error': f"Export failed: {str(e)}",
                'data': None
            }
    
    def get_available_features(self) -> Dict[str, Any]:
        """Get information about available raster features"""
        try:
            feature_info = self.model.get_feature_info()
            raster_paths = self._get_raster_paths()
            
            # Add availability status for each feature
            availability = {}
            feature_path_mapping = {
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
            
            for feature_name, path_key in feature_path_mapping.items():
                p = raster_paths.get(path_key, '')
                availability[feature_name] = {
                    'available': path_key in raster_paths and _path_or_url_exists(p),
                    'path_configured': path_key in raster_paths,
                    'file_exists': _path_or_url_exists(p) if path_key in raster_paths else False
                }
            
            return {
                'success': True,
                'features': feature_info['features'],
                'availability': availability,
                'metadata': {
                    'total_features': feature_info['total_features'],
                    'nodata_values': feature_info['nodata_values'],
                    'coordinate_system': feature_info['coordinate_system'],
                    'soil_databases_loaded': feature_info['soil_databases_loaded']
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting feature info: {e}")
            return {
                'success': False,
                'error': f"Failed to get feature info: {str(e)}",
                'features': None
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get service processing statistics"""
        return {
            'success': True,
            'statistics': {
                **self.processing_stats,
                'success_rate': (
                    self.processing_stats['successful_extractions'] / 
                    max(1, self.processing_stats['total_extractions'])
                ) * 100,
                'service_status': self._determine_service_status()
            }
        }
    
    def validate_raster_configuration(self) -> Dict[str, Any]:
        """Validate raster data configuration"""
        try:
            raster_paths = self._get_raster_paths()
            validation_results = {}
            
            if not raster_paths:
                return {
                    'success': False,
                    'error': "No raster paths configured",
                    'validation': None
                }
            
            for data_type, file_path in raster_paths.items():
                validation_results[data_type] = {
                    'path': file_path,
                    'exists': _path_or_url_exists(file_path),
                    'readable': False,
                    'is_remote': _is_url(file_path),
                    'error': None
                }
                
                if validation_results[data_type]['exists']:
                    try:
                        # Try to open the raster file
                        import rasterio
                        with rasterio.open(file_path) as src:
                            validation_results[data_type]['readable'] = True
                            validation_results[data_type]['crs'] = str(src.crs)
                            validation_results[data_type]['shape'] = src.shape
                    except Exception as e:
                        validation_results[data_type]['error'] = str(e)
            
            # Check soil database configuration
            soil_db_status = {
                'hwsd2_smu_configured': 'hwsd2_smu_path' in self.raster_config,
                'hwsd2_wrb4_configured': 'hwsd2_wrb4_path' in self.raster_config,
                'databases_loaded': self.model.soil_databases_loaded
            }
            
            return {
                'success': True,
                'validation': validation_results,
                'soil_databases': soil_db_status,
                'summary': {
                    'total_sources': len(raster_paths),
                    'available_sources': sum(1 for v in validation_results.values() if v['exists']),
                    'readable_sources': sum(1 for v in validation_results.values() if v['readable'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating raster configuration: {e}")
            return {
                'success': False,
                'error': f"Configuration validation failed: {str(e)}",
                'validation': None
            }
    
    def _determine_service_status(self) -> str:
        """Determine service health status based on processing statistics"""
        total_extractions = self.processing_stats['total_extractions'] 
        failed_extractions = self.processing_stats['failed_extractions']
        successful_extractions = self.processing_stats['successful_extractions']
        
        # If no extractions have been attempted, service is healthy (ready to use)
        if total_extractions == 0:
            return 'healthy'
        
        # If more than 50% fail, service is degraded
        if total_extractions > 0:
            failure_rate = failed_extractions / total_extractions
            if failure_rate > 0.5:
                return 'degraded'
        
        # Otherwise, service is healthy
        return 'healthy'
    
    def _get_raster_paths(self) -> Dict[str, str]:
        """Get configured raster file paths"""
        return self.raster_config.get('raster_paths', {})
    
    def test_raster_extraction(self) -> Dict[str, Any]:
        """Test raster extraction with sample coordinates"""
        try:
            # Use a sample coordinate (Philippines - safe for testing)
            test_coords = [{'longitude': 121.0, 'latitude': 14.0}]
            
            result = self.process_raster_extraction(test_coords)
            
            if result['success']:
                return {
                    'success': True,
                    'message': "Raster extraction test successful",
                    'test_data': result['data'][0] if result['data'] else None,
                    'processing_time': result['metadata']['processing_time_seconds']
                }
            else:
                return {
                    'success': False,
                    'message': "Raster extraction test failed",
                    'error': result['error']
                }
                
        except Exception as e:
            logger.error(f"Error in raster extraction test: {e}")
            return {
                'success': False,
                'message': "Raster extraction test failed",
                'error': str(e)
            }