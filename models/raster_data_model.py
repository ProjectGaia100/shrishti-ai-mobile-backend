"""
Raster Data Model for HazardGuard System
Extracts 9 geospatial features from COG-optimized raster data sources:
1. Soil type (HWSD2) - 33 soil classifications with database lookup         [soil_type.tif]
2. Elevation (WorldClim) - meters above sea level                           [elevation.tif]
3. Population density (GlobPOP) - persons per km²                           [population_density.tif]
4. Land cover (Copernicus) - 22 land cover classes                          [land_cover.tif]
5. NDVI (MODIS/eVIIRS) - Normalized Difference Vegetation Index             [ndvi.tif]
6. Annual precipitation (WorldClim) - mm per year                           [annual_precip.tif]
7. Annual mean temperature (WorldClim) - °C                                 [mean_annual_temp.tif]
8. Mean wind speed (Global Wind Atlas) - m/s                                [wind_speed.tif]
9. Impervious surface (GHSL) - percentage                                   [impervious_surface.tif]

All rasters are Cloud Optimized GeoTIFF (COG) with ZSTD compression, 256x256 tiles.
Data is 100% lossless — identical pixel values to original sources.
Files are served from GCS bucket (satellite-cog-data-for-shrishti) or local fallback
"""

import pandas as pd
import numpy as np
import os
import rasterio
from rasterio.warp import transform
import pyproj
import logging
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Note: PROJ/GDAL environment setup is done in main.py before any imports

class RasterDataModel:
    """Core model for raster data extraction and processing"""
    
    def __init__(self):
        """Initialize raster data model"""
        self.soil_databases_loaded = False
        self.smu_df = None
        self.wrb4_lookup = None
        
        # Soil type classification mapping (0-33)
        self.soil_classes = {
            'Acrisols': 1, 'Alisols': 2, 'Andosols': 3, 'Arenosols': 4, 'Calcisols': 5,
            'Cambisols': 6, 'Chernozems': 7, 'Ferralsols': 8, 'Fluvisols': 9, 'Gleysols': 10,
            'Gypsisols': 11, 'Histosols': 12, 'Kastanozems': 13, 'Leptosols': 14, 'Lixisols': 15,
            'Luvisols': 16, 'Nitisols': 17, 'Phaeozems': 18, 'Planosols': 19, 'Podzols': 20,
            'Regosols': 21, 'Solonchaks': 22, 'Solonetz': 23, 'Vertisols': 24, 'Unknown': 0,
            # Singular forms
            'Acrisol': 1, 'Alisol': 2, 'Andosol': 3, 'Arenosol': 4, 'Calcisol': 5,
            'Cambisol': 6, 'Chernozem': 7, 'Ferralsol': 8, 'Fluvisol': 9, 'Gleysol': 10,
            'Gypsisol': 11, 'Histosol': 12, 'Kastanozem': 13, 'Leptosol': 14, 'Lixisol': 15,
            'Luvisol': 16, 'Nitisol': 17, 'Phaeozem': 18, 'Planosol': 19, 'Podzol': 20,
            'Regosol': 21, 'Solonchak': 22, 'Solonetz': 23, 'Vertisol': 24,
            # Additional soil types
            'Anthrosols': 25, 'Cryosols': 26, 'Durisols': 27, 'Ferrasols': 28, 'Plinthosols': 29,
            'Retisols': 30, 'Stagnosols': 31, 'Technosols': 32, 'Umbrisols': 33
        }
        
        # Land cover classification mapping (0-21)
        self.land_cover_classes = {
            0: 0,    # Unknown (NoData)
            20: 1,   # Shrubs
            30: 2,   # Herbaceous vegetation
            40: 3,   # Cropland
            50: 4,   # Urban / built up
            60: 5,   # Bare / sparse vegetation
            70: 6,   # Snow and ice
            80: 7,   # Permanent water bodies
            90: 8,   # Herbaceous wetland
            100: 9,  # Moss and lichen
            111: 10, # Closed forest, evergreen needle leaf
            112: 11, # Closed forest, evergreen broad leaf
            113: 12, # Closed forest, deciduous needle leaf
            114: 13, # Closed forest, deciduous broad leaf
            115: 14, # Closed forest, mixed
            116: 15, # Closed forest, unknown
            121: 16, # Open forest, evergreen needle leaf
            122: 17, # Open forest, evergreen broad leaf
            123: 18, # Open forest, deciduous needle leaf
            124: 19, # Open forest, deciduous broad leaf
            125: 20, # Open forest, mixed
            126: 21  # Open forest, unknown
        }
    
    def load_soil_databases(self, hwsd2_path: str, wrb4_path: str) -> bool:
        """Load HWSD2 SMU and WRB4 lookup tables"""
        try:
            self.smu_df = pd.read_excel(hwsd2_path, index_col='HWSD2_SMU_ID')
            wrb4_df = pd.read_excel(wrb4_path)
            self.wrb4_lookup = dict(zip(wrb4_df['CODE'], wrb4_df['VALUE']))
            self.soil_databases_loaded = True
            
            logger.info(f"Loaded {len(self.smu_df)} SMU records and {len(self.wrb4_lookup)} WRB4 codes")
            return True
            
        except Exception as e:
            logger.error(f"Error loading soil databases: {e}")
            self.soil_databases_loaded = False
            return False
    
    def encode_soil_class(self, soil_class_name: str) -> int:
        """Encode soil class name to integer (0-33)"""
        return self.soil_classes.get(soil_class_name, 0)
    
    def encode_land_cover(self, lc_value: int) -> int:
        """Encode Copernicus land cover classes (0-21)"""
        return self.land_cover_classes.get(lc_value, 0)
    
    def extract_soil_type(self, coords: List[Tuple[float, float]], raster_path: str) -> List[int]:
        """Extract soil type with database lookup"""
        if not self.soil_databases_loaded:
            logger.error("Soil databases not loaded")
            return [0] * len(coords)
        
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Soil Raster NoData: {src.nodata}")
                soil_smus = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), soil_smu in zip(coords, soil_smus):
                    if soil_smu == 65535 or soil_smu == src.nodata or pd.isna(soil_smu):
                        results.append(0)  # Unknown
                        logger.debug(f"NoData soil for lat={lat}, lon={lon}")
                    else:
                        try:
                            wrb4_code = self.smu_df.loc[int(soil_smu), 'WRB4']
                            if pd.isna(wrb4_code) or wrb4_code == '':
                                soil_class_name = 'Unknown'
                            else:
                                soil_class_name = self.wrb4_lookup.get(wrb4_code, 'Unknown')
                            
                            # Extract main soil class (e.g., "Haplic Acrisols" -> "Acrisols")
                            soil_main = soil_class_name.split()[-1] if len(soil_class_name.split()) > 1 else soil_class_name
                            
                            # Try main class first, then full name, then default to 0
                            soil_class_encoded = self.encode_soil_class(soil_main)
                            if soil_class_encoded == 0 and soil_main != soil_class_name:
                                soil_class_encoded = self.encode_soil_class(soil_class_name)
                            
                            results.append(soil_class_encoded)
                            logger.debug(f"Got soil type {soil_class_name} (main: {soil_main}, code {soil_class_encoded}) for lat={lat}, lon={lon}")
                            
                        except (KeyError, ValueError):
                            results.append(0)  # Unknown
                            logger.debug(f"Missing soil data for SMU {soil_smu} at lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in soil type extraction: {e}")
            return [0] * len(coords)
    
    def extract_elevation(self, coords: List[Tuple[float, float]], raster_path: str) -> List[float]:
        """Extract elevation in meters"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Elevation Raster NoData: {src.nodata}")
                elevations = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), elev in zip(coords, elevations):
                    if elev == src.nodata or pd.isna(elev):
                        results.append(-9999.0)
                        logger.debug(f"NoData elevation for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python float
                        elev_val = float(elev) if hasattr(elev, 'item') else float(elev)
                        results.append(round(elev_val, 2))
                        logger.debug(f"Got elevation {elev_val:.2f}m for lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in elevation extraction: {e}")
            return [-9999.0] * len(coords)
    
    def extract_population_density(self, coords: List[Tuple[float, float]], raster_path: str) -> List[float]:
        """Extract population density in persons/km²"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Population Raster NoData: {src.nodata}")
                populations = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), pop in zip(coords, populations):
                    if pop == src.nodata or pd.isna(pop):
                        results.append(-9999.0)
                        logger.debug(f"NoData population for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python float
                        pop_val = float(pop) if hasattr(pop, 'item') else float(pop)
                        results.append(round(pop_val, 2))
                        logger.debug(f"Got population density {pop_val:.2f} persons/km² for lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in population extraction: {e}")
            return [-9999.0] * len(coords)
    
    def extract_land_cover(self, coords: List[Tuple[float, float]], raster_path: str) -> List[int]:
        """Extract land cover classification"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Land Cover Raster NoData: {src.nodata}")
                landcovers = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), lc_code in zip(coords, landcovers):
                    if lc_code == src.nodata or pd.isna(lc_code) or lc_code not in self.land_cover_classes:
                        logger.debug(f"NoData or invalid land cover for lat={lat}, lon={lon}")
                        results.append(0)  # Default to 0 (Unknown)
                    else:
                        lc_encoded = self.land_cover_classes[int(lc_code)]
                        logger.debug(f"Got land cover class {lc_encoded} (code: {lc_code}) for lat={lat}, lon={lon}")
                        results.append(lc_encoded)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in land cover extraction: {e}")
            return [0] * len(coords)
    
    def extract_ndvi(self, coords: List[Tuple[float, float]], raster_path: str) -> List[float]:
        """Extract NDVI with scaling factor /10000"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"NDVI Raster NoData: {src.nodata}")
                ndvi_values = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), ndvi_val in zip(coords, ndvi_values):
                    if ndvi_val == -9999.0 or ndvi_val == src.nodata or pd.isna(ndvi_val):
                        results.append(-9999.0)
                        logger.debug(f"NoData NDVI for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python float
                        ndvi_raw = float(ndvi_val) if hasattr(ndvi_val, 'item') else float(ndvi_val)
                        scaled_ndvi = ndvi_raw / 10000.0
                        rounded_ndvi = round(scaled_ndvi, 4)
                        results.append(rounded_ndvi)
                        logger.debug(f"Got NDVI {rounded_ndvi} for lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in NDVI extraction: {e}")
            return [-9999.0] * len(coords)
    
    def extract_annual_precipitation(self, coords: List[Tuple[float, float]], raster_path: str) -> List[int]:
        """Extract annual precipitation in mm"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Precip Raster NoData: {src.nodata}")
                precips = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), precip in zip(coords, precips):
                    if precip == src.nodata or pd.isna(precip):
                        results.append(-9999)
                        logger.debug(f"NoData precip for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python int
                        precip_val = float(precip) if hasattr(precip, 'item') else float(precip)
                        rounded_precip = int(round(precip_val, 0))
                        results.append(rounded_precip)
                        logger.debug(f"Got annual precip {rounded_precip} mm for lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in precipitation extraction: {e}")
            return [-9999] * len(coords)
    
    def extract_annual_temperature(self, coords: List[Tuple[float, float]], raster_path: str) -> List[float]:
        """Extract annual mean temperature in °C"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Temp Raster NoData: {src.nodata}")
                temps = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), temp in zip(coords, temps):
                    if temp == src.nodata or pd.isna(temp):
                        results.append(-9999.0)
                        logger.debug(f"NoData temp for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python float
                        temp_val = float(temp) if hasattr(temp, 'item') else float(temp)
                        rounded_temp = round(temp_val, 1)
                        results.append(rounded_temp)
                        logger.debug(f"Got annual mean temp {rounded_temp} °C for lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in temperature extraction: {e}")
            return [-9999.0] * len(coords)
    
    def extract_wind_speed(self, coords: List[Tuple[float, float]], raster_path: str) -> List[float]:
        """Extract mean wind speed in m/s"""
        try:
            with rasterio.open(raster_path) as src:
                logger.debug(f"Wind Raster NoData: {src.nodata}")
                winds = [val[0] for val in src.sample(coords)]
                results = []
                
                for (lon, lat), wind in zip(coords, winds):
                    if wind == src.nodata or pd.isna(wind):
                        results.append(-9999.0)
                        logger.debug(f"NoData wind for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python float
                        wind_val = float(wind) if hasattr(wind, 'item') else float(wind)
                        rounded_wind = round(wind_val, 2)
                        results.append(rounded_wind)
                        logger.debug(f"Got mean wind speed {rounded_wind} m/s for lat={lat}, lon={lon}")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in wind speed extraction: {e}")
            return [-9999.0] * len(coords)
    
    def extract_impervious_surface(self, coords: List[Tuple[float, float]], raster_path: str) -> List[float]:
        """Extract impervious surface percentage with CRS transformation"""
        try:
            # Check if raster file exists (skip check for URLs — rasterio handles them)
            is_url = raster_path.startswith('http://') or raster_path.startswith('https://')
            if not is_url and not os.path.exists(raster_path):
                logger.error(f"Impervious surface raster file not found: {raster_path}")
                return [-9999.0] * len(coords)
            
            with rasterio.open(raster_path) as src:
                logger.info(f"[IMPERVIOUS] Raster CRS: {src.crs}, NoData: {src.nodata}, dtype: {src.dtypes[0]}")
                
                # Transform coordinates from EPSG:4326 to raster's CRS (Mollweide, ESRI:54009)
                # Use pyproj.Transformer directly - more reliable than rasterio.warp.transform
                # because pyproj manages its own proj.db path independently
                lons = [lon for lon, lat in coords]
                lats = [lat for lon, lat in coords]
                
                try:
                    transformer = pyproj.Transformer.from_crs(
                        'EPSG:4326', src.crs.to_string(), always_xy=True
                    )
                    transformed_lons, transformed_lats = transformer.transform(lons, lats)
                    transformed_coords = list(zip(transformed_lons, transformed_lats))
                    for i, (lon, lat) in enumerate(coords):
                        logger.info(f"[IMPERVIOUS] CRS transform: ({lon}, {lat}) -> ({transformed_lons[i]:.1f}, {transformed_lats[i]:.1f})")
                except Exception as transform_error:
                    logger.error(f"Coordinate transformation failed: {transform_error}")
                    return [-9999.0] * len(coords)
                
                impervs = [val[0] for val in src.sample(transformed_coords)]
                results = []
                
                for (lon, lat), imperv in zip(coords, impervs):
                    if imperv == src.nodata or pd.isna(imperv):
                        results.append(-9999.0)  # Standard NoData for impervious
                        logger.info(f"[IMPERVIOUS] NoData (={src.nodata}) for lat={lat}, lon={lon}")
                    else:
                        # Convert numpy types to native Python float
                        imperv_val = float(imperv) if hasattr(imperv, 'item') else float(imperv)
                        # Apply scaling factor for GHSL (divide by 100)
                        scaled_imperv = imperv_val / 100.0
                        # Round to 2 decimal places (percentage)
                        rounded_imperv = round(scaled_imperv, 2)
                        results.append(rounded_imperv)
                        logger.info(f"[IMPERVIOUS] lat={lat}, lon={lon} -> raw={int(imperv_val)}, scaled={rounded_imperv}%")
                
                return results
                
        except rasterio.errors.RasterioIOError as io_error:
            logger.error(f"Rasterio I/O error in impervious surface extraction: {io_error}")
            return [-9999.0] * len(coords)
        except Exception as e:
            logger.error(f"Error in impervious surface extraction: {e}")
            logger.error(f"Raster path: {raster_path}")
            return [-9999.0] * len(coords)
    
    def extract_all_features(self, coords: List[Tuple[float, float]], raster_paths: Dict[str, str]) -> Dict[str, List[Any]]:
        """Extract all 9 raster features in a single operation"""
        logger.info(f"Extracting all raster features for {len(coords)} coordinates")
        
        # Setup PROJ paths for all raster operations (handles Flask reloader)
        # Use environment variables set by main.py, with fallback auto-detection
        from pathlib import Path
        
        proj_lib = os.environ.get('PROJ_LIB', '')
        gdal_data = os.environ.get('GDAL_DATA', '')
        
        # Fallback: try common locations if env vars are not set (cross-platform)
        if not proj_lib:
            candidates = []
            try:
                import rasterio as _rio
                candidates.append(Path(_rio.__file__).parent / "proj_data")
            except ImportError:
                pass
            try:
                import pyproj as _pp
                candidates.append(Path(_pp.datadir.get_data_dir()))
            except (ImportError, AttributeError):
                pass
            candidates.extend([Path("/usr/share/proj"), Path("/usr/local/share/proj")])
            for c in candidates:
                if c.exists() and (c / "proj.db").exists():
                    proj_lib = str(c)
                    break
        
        if not gdal_data:
            candidates = []
            try:
                from osgeo import gdal as _gdal
                candidates.append(Path(_gdal.__file__).parent / "data" / "gdal")
                candidates.append(Path(_gdal.__file__).parent / "data")
            except ImportError:
                pass
            candidates.extend([Path("/usr/share/gdal"), Path("/usr/local/share/gdal")])
            for c in candidates:
                if c.exists():
                    gdal_data = str(c)
                    break
        
        # Suppress noisy PROJ "Cannot find proj.db" warnings from rasterio/GDAL
        # These are harmless for rasters that don't need CRS transformation
        rasterio_logger = logging.getLogger('rasterio._env')
        original_level = rasterio_logger.level
        rasterio_logger.setLevel(logging.CRITICAL)
        
        try:
            # Wrap ALL raster operations in PROJ environment
            env_kwargs = {'PROJ_IGNORE_CELESTIAL_BODY': '1'}
            if proj_lib:
                env_kwargs['PROJ_LIB'] = proj_lib
            if gdal_data:
                env_kwargs['GDAL_DATA'] = gdal_data
            
            with rasterio.Env(**env_kwargs):
                return self._extract_all_features_internal(coords, raster_paths)
        finally:
            rasterio_logger.setLevel(original_level)
    
    def _extract_all_features_internal(self, coords: List[Tuple[float, float]], raster_paths: Dict[str, str]) -> Dict[str, List[Any]]:
        """Internal method for feature extraction (called within PROJ environment)"""
        results = {}
        
        # Extract soil type
        if 'soil' in raster_paths and self.soil_databases_loaded:
            results['soil_type'] = self.extract_soil_type(coords, raster_paths['soil'])
        else:
            results['soil_type'] = [0] * len(coords)
            logger.warning("Soil data not available or databases not loaded")
        
        # Extract elevation
        if 'elevation' in raster_paths:
            results['elevation_m'] = self.extract_elevation(coords, raster_paths['elevation'])
        else:
            results['elevation_m'] = [-9999.0] * len(coords)
            logger.warning("Elevation data not available")
        
        # Extract population density
        if 'population' in raster_paths:
            results['pop_density_persqkm'] = self.extract_population_density(coords, raster_paths['population'])
        else:
            results['pop_density_persqkm'] = [-9999.0] * len(coords)
            logger.warning("Population data not available")
        
        # Extract land cover
        if 'landcover' in raster_paths:
            results['land_cover_class'] = self.extract_land_cover(coords, raster_paths['landcover'])
        else:
            results['land_cover_class'] = [0] * len(coords)
            logger.warning("Land cover data not available")
        
        # Extract NDVI
        if 'ndvi' in raster_paths:
            results['ndvi'] = self.extract_ndvi(coords, raster_paths['ndvi'])
        else:
            results['ndvi'] = [-9999.0] * len(coords)
            logger.warning("NDVI data not available")
        
        # Extract annual precipitation
        if 'precip' in raster_paths:
            results['annual_precip_mm'] = self.extract_annual_precipitation(coords, raster_paths['precip'])
        else:
            results['annual_precip_mm'] = [-9999] * len(coords)
            logger.warning("Precipitation data not available")
        
        # Extract annual temperature
        if 'temp' in raster_paths:
            results['annual_mean_temp_c'] = self.extract_annual_temperature(coords, raster_paths['temp'])
        else:
            results['annual_mean_temp_c'] = [-9999.0] * len(coords)
            logger.warning("Temperature data not available")
        
        # Extract wind speed
        if 'wind' in raster_paths:
            results['mean_wind_speed_ms'] = self.extract_wind_speed(coords, raster_paths['wind'])
        else:
            results['mean_wind_speed_ms'] = [-9999.0] * len(coords)
            logger.warning("Wind data not available")
        
        # Extract impervious surface
        if 'impervious' in raster_paths:
            results['impervious_surface_pct'] = self.extract_impervious_surface(coords, raster_paths['impervious'])
        else:
            results['impervious_surface_pct'] = [-9999.0] * len(coords)
            logger.warning("Impervious surface data not available")
        
        logger.info(f"Successfully extracted all raster features for {len(coords)} coordinates")
        return results
    
    def validate_coordinates(self, coords: List[Tuple[float, float]]) -> bool:
        """Validate coordinate format and ranges"""
        try:
            for lon, lat in coords:
                # Check if coordinates are numeric
                if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                    return False
                
                # Check coordinate ranges
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available raster features"""
        return {
            'features': {
                'soil_type': {
                    'description': 'Soil classification (HWSD2)',
                    'range': '0-33 (encoded classes)',
                    'classes': len(self.soil_classes),
                    'unit': 'categorical'
                },
                'elevation_m': {
                    'description': 'Elevation above sea level',
                    'range': 'varies by location',
                    'unit': 'meters'
                },
                'pop_density_persqkm': {
                    'description': 'Population density',
                    'range': '0-∞',
                    'unit': 'persons/km²'
                },
                'land_cover_class': {
                    'description': 'Land cover classification (Copernicus)',
                    'range': '0-21 (encoded classes)',
                    'classes': len(self.land_cover_classes),
                    'unit': 'categorical'
                },
                'ndvi': {
                    'description': 'Normalized Difference Vegetation Index',
                    'range': '-1.0 to 1.0',
                    'unit': 'index'
                },
                'annual_precip_mm': {
                    'description': 'Annual precipitation',
                    'range': '0-∞',
                    'unit': 'mm/year'
                },
                'annual_mean_temp_c': {
                    'description': 'Annual mean temperature',
                    'range': 'varies by location',
                    'unit': '°C'
                },
                'mean_wind_speed_ms': {
                    'description': 'Mean wind speed',
                    'range': '0-∞',
                    'unit': 'm/s'
                },
                'impervious_surface_pct': {
                    'description': 'Impervious surface coverage',
                    'range': '0-100',
                    'unit': 'percentage'
                }
            },
            'total_features': 9,
            'nodata_values': {
                'numeric': -9999.0,
                'categorical': 0
            },
            'coordinate_system': 'EPSG:4326 (WGS84)',
            'soil_databases_loaded': self.soil_databases_loaded
        }