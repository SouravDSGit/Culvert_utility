# ================================================================================================
# 📊 Optimized WDFM Utility Functions - Final Robust Implementation
# ================================================================================================

import numpy as np
import pandas as pd
import tempfile
import glob
import shutil
import warnings
import os
import traceback
import uuid
from typing import Optional, Tuple, List, Dict, Any, Union

# ================================================================================================
# 🧪 Geospatial Analysis
# ================================================================================================
import geopandas as gpd
from shapely.geometry import mapping, box, Polygon, MultiPolygon
from shapely.ops import unary_union
import whitebox

# ================================================================================================
# 🗺️ Raster & Vector Data Handling
# ================================================================================================
import rasterio
from rasterio import features, mask as rio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from rasterio.windows import from_bounds, Window
from osgeo import gdal, ogr, osr
from pyproj import CRS
from affine import Affine

# ================================================================================================
# 🧮 Scientific & Statistical Functions
# ================================================================================================
from scipy.ndimage import distance_transform_edt

# ================================================================================================
# ⚙️ Configuration and Memory Management
# ================================================================================================
gdal.SetCacheMax(256 * 1024 * 1024)
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
gdal.SetConfigOption('GDAL_TIFF_OVR_BLOCKSIZE', '512')

wbt = whitebox.WhiteboxTools()

# ================================================================================================
# 🔧 Core Helper Functions (Proven Robust Version)
# ================================================================================================
def create_dissolved_boundary_shapefile(boundary_path: str, output_path: str, target_crs: Any = None) -> str:
    """Creates a dissolved, single-polygon, and topologically valid boundary shapefile."""
    boundary_gdf = gpd.read_file(boundary_path)
    if len(boundary_gdf) == 0:
        raise ValueError("Input boundary shapefile is empty")
    print(f"  Loading boundary: {len(boundary_gdf)} features")
    if target_crs and boundary_gdf.crs != target_crs:
        boundary_gdf = boundary_gdf.to_crs(target_crs)
    dissolved_geom = boundary_gdf.unary_union
    if not dissolved_geom.is_valid:
        dissolved_geom = dissolved_geom.buffer(0)
    if dissolved_geom.geom_type == 'GeometryCollection':
        polygons = [geom for geom in dissolved_geom.geoms if geom.geom_type in ('Polygon', 'MultiPolygon')]
        if not polygons:
            raise ValueError("GeometryCollection does not contain any polygons after dissolving.")
        dissolved_geom = unary_union(polygons)
    dissolved_gdf = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=boundary_gdf.crs)
    if not dissolved_gdf.geometry.iloc[0].is_valid:
        raise ValueError("Failed to create a valid dissolved boundary geometry.")
    dissolved_gdf.to_file(output_path, driver='ESRI Shapefile')
    print(f"  ✓ Geometry validated: {dissolved_gdf.geometry.iloc[0].geom_type}")
    print(f"  ✓ Created dissolved boundary: 1 polygon from {len(boundary_gdf)} features")
    return output_path

def create_template_from_boundary_and_dem(boundary_path: str, dem_path: str,
                                         output_template_path: str) -> Dict[str, Any]:
    """Creates a raster template by clipping a DEM to a boundary, preserving its native resolution and CRS."""
    print("Creating resolution-preserving template from DEM...")
    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        print(f"  DEM CRS: {dem_crs}")
        print(f"  DEM native resolution: {abs(dem_src.res[0]):.2f} x {abs(dem_src.res[1]):.2f} m")
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_gdf.crs != dem_crs:
            boundary_gdf = boundary_gdf.to_crs(dem_crs)
        geoms_transformed = [mapping(geom) for geom in boundary_gdf.geometry]
        masked_array, clipped_transform = rio_mask.mask(
            dem_src, geoms_transformed, crop=True, all_touched=True
        )
        clipped_data = np.ma.filled(masked_array.astype('float32'), fill_value=np.nan)
        template_height, template_width = clipped_data.shape[1], clipped_data.shape[2]
        if template_width <= 0 or template_height <= 0:
            raise ValueError(f"Template creation resulted in invalid dimensions ({template_width}x{template_height}).")
        template_bounds = rasterio.transform.array_bounds(template_height, template_width, clipped_transform)
        actual_resolution = (clipped_transform.a, abs(clipped_transform.e))
        print(f"  Template dimensions: {template_width} x {template_height}")
        print(f"  Template resolution: {actual_resolution[0]:.2f} x {actual_resolution[1]:.2f} m")
        template_meta = dem_src.meta.copy()
        template_meta.update({
            'height': template_height, 'width': template_width, 'transform': clipped_transform,
            'nodata': np.nan, 'dtype': 'float32', 'driver': 'GTiff', 'compress': 'lzw'
        })
        with rasterio.open(output_template_path, 'w', **template_meta) as dst:
            dst.write(clipped_data)
        return {
            'path': output_template_path, 'transform': clipped_transform, 'width': template_width,
            'height': template_height, 'bounds': template_bounds, 'crs': dem_crs, 'resolution': actual_resolution
        }

def process_raster_to_template(input_path: str, output_path: str, template_props: Dict[str, Any],
                                boundary_path: str, resampling_method: Resampling = Resampling.bilinear,
                                temp_dir: str = None, input_crs_override: str = None) -> str:
    """Robustly processes any raster to match template properties using a safe, multi-step clip-then-warp approach."""
    print(f"Processing raster to template: {os.path.basename(input_path)}")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    try:
        info = gdal.Info(input_path, format='json')
        src_crs = CRS.from_wkt(info['coordinateSystem']['wkt']) if 'coordinateSystem' in info else None
        if input_crs_override and src_crs is None:
             src_crs = CRS.from_string(input_crs_override)
        if src_crs is None:
            raise ValueError(f"Source CRS for {os.path.basename(input_path)} could not be determined.")
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_gdf.crs != src_crs:
            boundary_gdf = boundary_gdf.to_crs(src_crs)
        geoms_for_clipping = [mapping(geom) for geom in boundary_gdf.geometry]
        temp_clipped_path = os.path.join(local_temp_dir, "clipped.tif")
        with rasterio.open(input_path) as src:
            masked_array, clipped_transform = rio_mask.mask(
                src, geoms_for_clipping, crop=True, all_touched=True
            )
            clipped_data_float = np.ma.filled(masked_array.astype('float32'), fill_value=np.nan)
            if src.nodata is not None:
                clipped_data_float[clipped_data_float == src.nodata] = np.nan
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                'driver': 'GTiff', 'compress': 'lzw', 'height': clipped_data_float.shape[1],
                'width': clipped_data_float.shape[2], 'transform': clipped_transform,
                'nodata': np.nan, 'dtype': 'float32'
            })
            with rasterio.open(temp_clipped_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_data_float)
        gdal_resampling_map = {
            Resampling.nearest: 'near', Resampling.bilinear: 'bilinear',
            Resampling.cubic: 'cubic', Resampling.mode: 'mode', Resampling.average: 'average'
        }
        warp_options = gdal.WarpOptions(
            format='GTiff', dstSRS=template_props['crs'].to_wkt(),
            outputBounds=template_props['bounds'], width=template_props['width'], height=template_props['height'],
            resampleAlg=gdal_resampling_map.get(resampling_method, 'bilinear'),
            dstNodata=np.nan, creationOptions=['COMPRESS=LZW', 'TILED=YES'], multithread=True,
            srcNodata=np.nan
        )
        ds = gdal.Warp(output_path, temp_clipped_path, options=warp_options)
        if ds is None:
            raise RuntimeError(f"GDAL Warp failed after clipping. Last error: {gdal.GetLastErrorMsg()}")
        ds = None
        output_size = os.path.getsize(output_path)
        print(f"  ✓ Warp completed (size: {output_size} bytes)")
        return output_path
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to process {os.path.basename(input_path)}: {str(e)}")
    finally:
        shutil.rmtree(local_temp_dir)

def cleanup_shapefile(shapefile_path: str):
    """Clean up all components of a shapefile."""
    base_path = os.path.splitext(shapefile_path)[0]
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml']:
        file_path = base_path + ext
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

def categorize_raster_data(input_data: np.ndarray, category_type: str,
                          thresholds: Optional[List[float]] = None,
                          mapping_dict: Optional[Dict[Any, int]] = None) -> np.ndarray:
    """Categorizes raster data using predefined rules, ensuring float output."""
    output = np.full_like(input_data, np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(input_data)
    if not np.any(valid_mask):
        return output
    if category_type == 'range' and thresholds is not None:
        conditions = [input_data <= thresholds[0]]
        for i in range(1, len(thresholds)):
            conditions.append((input_data > thresholds[i-1]) & (input_data <= thresholds[i]))
        conditions.append(input_data > thresholds[-1])
        choices = list(range(1, len(thresholds) + 2))
        categorized = np.select(conditions, choices, default=np.nan)
        output[valid_mask] = categorized[valid_mask]
    elif category_type == 'categorical' and mapping_dict is not None:
        for original_val, category_val in mapping_dict.items():
            mask = (input_data == original_val) & valid_mask
            output[mask] = category_val
    return output

def get_us_states_crossed(polygon_path: str, usa_states_shapefile_path: str,
                         state_abbr_column: str = "stusps") -> List[str]:
    """Finds the US state abbreviations that the given boundary polygon crosses."""
    try:
        boundary_gdf = gpd.read_file(polygon_path).to_crs("EPSG:4326")
        states_gdf = gpd.read_file(f"zip://{usa_states_shapefile_path}").to_crs("EPSG:4326")
        boundary_union = boundary_gdf.geometry.unary_union
        intersecting_states = states_gdf[states_gdf.intersects(boundary_union)]
        if intersecting_states.empty or state_abbr_column not in intersecting_states.columns:
            return []
        return intersecting_states[state_abbr_column].unique().tolist()
    except Exception as e:
        print(f"Error finding intersecting states: {e}")
        return []

def create_nan_raster_from_template(template_path: str, output_path: str):
    """Helper function to create a NaN raster matching template specifications."""
    with rasterio.open(template_path) as template:
        nan_data = np.full((template.height, template.width), np.nan, dtype=np.float32)
        out_meta = template.meta.copy()
        out_meta.update({'dtype': 'float32', 'nodata': np.nan})
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(nan_data, 1)

# ===================================================================================================================
# ====================== WDFM Specific Data Processing Workflows ====================================================
# ===================================================================================================================

def calculate_and_categorize_slope(dem_path: str, boundary_shapefile_path: str,
                                  slope_percentage_path: str, categorized_slope_path: str,
                                  temp_dir: str = None) -> Tuple[str, str, List[float]]:
    """Memory-efficient slope calculation and categorization using GDAL."""
    print("Starting optimized slope calculation and categorization...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_slope.shp")
        create_dissolved_boundary_shapefile(boundary_shapefile_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem.tif")
        create_template_from_boundary_and_dem(dissolved_boundary_path, dem_path, template_path)
        print("Calculating slope using GDAL...")
        slope_options = gdal.DEMProcessingOptions(
            format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES'], slopeFormat='percent',
            computeEdges=True, alg='ZevenbergenThorne'
        )
        gdal.DEMProcessing(slope_percentage_path, template_path, 'slope', options=slope_options)
        print("Categorizing slope...")
        with rasterio.open(slope_percentage_path) as slope_src:
            slope_data = slope_src.read(1)
            slope_data[slope_data < 0] = np.nan
            thresholds = [5, 10, 20, 45] # 4 thresholds for 5 categories
            categorized_data = categorize_raster_data(slope_data, 'range', thresholds=thresholds)
            slope_meta = slope_src.meta.copy()
            slope_meta.update({'dtype': 'float32', 'nodata': np.nan})
            with rasterio.open(categorized_slope_path, 'w', **slope_meta) as dst:
                dst.write(categorized_data.astype(np.float32), 1)
        print("✓ Slope calculation and categorization completed successfully!")
        return slope_percentage_path, categorized_slope_path, thresholds
    except Exception as e:
        raise RuntimeError(f"Slope calculation failed: {str(e)}")
    finally:
        shutil.rmtree(local_temp_dir, ignore_errors=True)

def process_noaa_atlas14_for_boundary(atlas14_dir: str, boundary_shapefile: str, dem_path: str,
                                      output_raster_path: str, categorized_raster_path: str = None,
                                      temp_dir: str = None, method_prefix: str = "wdfm") -> str:
    """Processes NOAA Atlas 14 data robustly using the standardized, template-based approach."""
    print("Starting optimized NOAA Atlas 14 processing...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_noaa.shp")
        create_dissolved_boundary_shapefile(boundary_shapefile, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem_noaa.tif")
        template_props = create_template_from_boundary_and_dem(dissolved_boundary_path, dem_path, template_path)
        asc_files = glob.glob(os.path.join(atlas14_dir, "**", "*.asc"), recursive=True)
        if not asc_files:
            raise ValueError("No ASC files found in the specified directory")
        boundary_gdf_wgs84 = gpd.read_file(dissolved_boundary_path).to_crs("EPSG:4326")
        boundary_for_intersect = boundary_gdf_wgs84.unary_union
        intersecting_files = []
        for asc_file in asc_files:
            try:
                with rasterio.open(asc_file) as src:
                    src_bounds_wgs84 = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    if boundary_for_intersect.intersects(box(*src_bounds_wgs84)):
                        intersecting_files.append(asc_file)
            except Exception:
                continue
        if not intersecting_files:
            raise ValueError("No ASC files intersect with the boundary area")
        processed_rasters = []
        for i, asc_file in enumerate(intersecting_files):
            temp_processed_path = os.path.join(local_temp_dir, f"processed_{i}.tif")
            try:
                process_raster_to_template(
                    asc_file, temp_processed_path, template_props, dissolved_boundary_path,
                    resampling_method=Resampling.bilinear, temp_dir=local_temp_dir
                )
                processed_rasters.append(temp_processed_path)
            except Exception as e:
                print(f"  Warning: Failed to process {os.path.basename(asc_file)}: {e}")
        if not processed_rasters:
            raise ValueError("No valid rasters were produced from NOAA data.")
        final_raster = processed_rasters[0]
        if len(processed_rasters) > 1:
            print(f"Merging {len(processed_rasters)} rasters...")
            final_raster = os.path.join(local_temp_dir, "merged.tif")
            vrt = gdal.BuildVRT(os.path.join(local_temp_dir, "merged.vrt"), processed_rasters)
            gdal.Translate(final_raster, vrt)
            vrt = None
        shutil.copy2(final_raster, output_raster_path)
        print(f"✓ Saved final raster: {output_raster_path}")
        if categorized_raster_path:
            print("Creating categorized version...")
            with rasterio.open(output_raster_path) as src:
                data = src.read(1)
                if np.any(~np.isnan(data)):
                    bins = np.linspace(np.nanmin(data), np.nanmax(data), 5)
                    categorized_data_int = np.digitize(data, bins, right=True)
                    categorized_data = categorized_data_int.astype('float32')
                    categorized_data[np.isnan(data)] = np.nan
                    cat_meta = src.meta.copy()
                    cat_meta.update({'dtype': 'float32', 'nodata': np.nan})
                    with rasterio.open(categorized_raster_path, 'w', **cat_meta) as dst:
                        dst.write(categorized_data, 1)
                    print(f"✓ Saved categorized raster: {categorized_raster_path}")
        return output_raster_path
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"NOAA Atlas 14 processing failed: {str(e)}")
    finally:
        shutil.rmtree(local_temp_dir, ignore_errors=True)

def gssurgo_to_wdfm_rasters(gssurgo_soil_data_directory_path: str, boundary_shp_path: str,
                           dem_path: str, usa_states_shapefile_path: str, output_dir: str,
                           temp_dir: str = None) -> List[str]:
    """
    Memory-efficient processing of GSSURGO soil data to WDFM categorized rasters,
    with correct handling for direct and inverse vulnerability scales.
    Now saves both raw and categorized versions.
    """
    print("Starting optimized GSSURGO to WDFM rasters processing...")
    os.makedirs(output_dir, exist_ok=True)
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_gssurgo.shp")
        create_dissolved_boundary_shapefile(boundary_shp_path, dissolved_boundary_path)
        
        template_path = os.path.join(local_temp_dir, "template_dem_gssurgo.tif")
        template_props = create_template_from_boundary_and_dem(dissolved_boundary_path, dem_path, template_path)
        
        state_abbrs = get_us_states_crossed(dissolved_boundary_path, usa_states_shapefile_path)
        if not state_abbrs:
            raise ValueError("Could not determine states for the boundary")
        print(f"Boundary crosses: {state_abbrs}")

        # --- REVISED WDFM VARIABLES DICTIONARY ---
        # Added 'inverse: True' flag to variables needing an inverted score.
        wdfm_variables = {
            'rootznaws': {'type': 'range', 'thresholds': [50, 100, 150, 200]},
            'drainagecl': {'type': 'categorical', 'mapping_dict': {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5}},
            'kwfact': {'type': 'range', 'thresholds': [0.13, 0.26, 0.39, 0.52]},
            'ksat_h': {
                'type': 'range', 
                'thresholds': [1, 5, 20, 50], 
                'inverse': True  # Low Ksat = High Risk
            },
            'runoff': {'type': 'categorical', 'mapping_dict': {0: 1, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}},
            'soilslippot': {'type': 'categorical', 'mapping_dict': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}},
            'resdepb_r': {
                'type': 'range', 
                'thresholds': [25, 50, 100, 150], 
                'inverse': True  # Low Depth = High Risk
            },
            'taxorder': {'type': 'categorical', 'mapping_dict': {1: 1, 2: 5, 3: 4, 4: 4, 5: 2, 6: 3, 7: 2, 8: 2, 9: 3, 10: 1, 11: 2, 12: 3}},
            'tfact': {'type': 'range', 'thresholds': [1, 2, 3, 4]}
        }
        
        output_paths = []
        for var_key, config in wdfm_variables.items():
            print(f"\nProcessing '{var_key}'")
            try:
                source_raster_paths = [p for state in state_abbrs if os.path.exists(p := os.path.join(gssurgo_soil_data_directory_path, state.upper(), f'{var_key}.tif'))]
                
                # Define both raw and categorized output paths
                raw_output_path = os.path.join(output_dir, f"raw_{var_key}_wdfm.tif")
                final_output_path = os.path.join(output_dir, f"{var_key}_wdfm.tif")
                
                if not source_raster_paths:
                    print(f"  No source rasters found, creating NaN rasters.")
                    create_nan_raster_from_template(template_path, raw_output_path)
                    create_nan_raster_from_template(template_path, final_output_path)
                    output_paths.append(final_output_path)
                    continue
                    
                input_for_proc = source_raster_paths[0]
                if len(source_raster_paths) > 1:
                    vrt_path = os.path.join(local_temp_dir, f"{var_key}.vrt")
                    gdal.BuildVRT(vrt_path, source_raster_paths)
                    input_for_proc = vrt_path
                
                # Process to template and save as RAW version
                print(f"  Saving raw clipped data to: raw_{var_key}_wdfm.tif")
                process_raster_to_template(
                    input_for_proc, raw_output_path, template_props, dissolved_boundary_path,
                    resampling_method=Resampling.nearest, temp_dir=local_temp_dir, input_crs_override='EPSG:5070'
                )
                
                # Read raw data and categorize
                with rasterio.open(raw_output_path) as src:
                    source_data = src.read(1)
                    categorized_data = categorize_raster_data(
                        source_data, 
                        config['type'], 
                        thresholds=config.get('thresholds'), 
                        mapping_dict=config.get('mapping_dict')
                    )
                    
                    # --- REVISED INVERSE LOGIC ---
                    # Check for the 'inverse' flag in the config dictionary.
                    if config.get('inverse', False):
                        print(f"  Inverting vulnerability scale for {var_key}...")
                        valid_mask = ~np.isnan(categorized_data)
                        # This formula correctly inverts a 1-5 scale:
                        # 1 -> 5, 2 -> 4, 3 -> 3, 4 -> 2, 5 -> 1
                        categorized_data[valid_mask] = 6 - categorized_data[valid_mask]

                    meta = src.meta.copy()
                    meta.update(dtype='float32', nodata=np.nan)
                    with rasterio.open(final_output_path, 'w', **meta) as dst:
                        dst.write(categorized_data, 1)
                
                print(f"  ✓ Raw data saved: {raw_output_path}")
                print(f"  ✓ Categorized data saved: {final_output_path}")
                output_paths.append(final_output_path)
            except Exception as e:
                print(f"  ✗ Error processing '{var_key}': {str(e)}")
                traceback.print_exc() # Print full traceback for debugging
                
        print(f"\n✓ GSSURGO processing complete: {len(output_paths)} rasters created")
        return output_paths
    except Exception as e:
        raise RuntimeError(f"GSSURGO processing failed: {str(e)}")
    finally:
        shutil.rmtree(local_temp_dir, ignore_errors=True)


def process_geology_to_raster(geology_dir_path: str, boundary_shp_path: str,
                             dem_path: str, categorized_geology_path: str,
                             usa_states_shapefile_path: str, temp_dir: str = None) -> str:
    """
    Memory-efficient processing of geology shapefile data to categorized raster.
    Now saves both raw clipped shapefile (with text rock types) and categorized raster.
    """
    print("Starting optimized geology processing...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_geology.shp")
        create_dissolved_boundary_shapefile(boundary_shp_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem_geology.tif")
        template_props = create_template_from_boundary_and_dem(dissolved_boundary_path, dem_path, template_path)
        
        geology_mapping = {
            'metasedimentary rock': 4, 'schist': 3, 'granitic gneiss': 2, 'gneiss': 3,
            'ultramafic intrusive rock': 1, 'biotite gneiss': 3, 'quartzite': 4, 'mica schist': 3,
            'slate': 4, 'limestone': 5, 'shale': 5, 'dolostone (dolomite)': 4, 'conglomerate': 3,
            'phyllite': 3, 'sandstone': 5, 'amphibolite': 3, 'water': 5, 'biotite schist': 4,
            'chert': 1, 'marble': 2, 'granite': 1, 'amphibole schist': 3, 'mylonite': 4,
            'felsic metavolcanic rock': 1, 'metavolcanic rock': 1, 'gabbro': 1, 'mafic metavolcanic rock': 1,
            'sand': 5, 'mafic gneiss': 3, 'alluvium': 4, 'clay or mud': 4, 'hornfels': 3,
            'charnockite': 4, 'dune sand': 5, 'unconsolidated deposit': 5, 'beach sand': 5, 'NODATA': np.nan
        }
        
        state_abbrs = get_us_states_crossed(dissolved_boundary_path, usa_states_shapefile_path)
        if not state_abbrs:
            print("Warning: Could not determine states for boundary, creating NaN raster.")
            create_nan_raster_from_template(template_path, categorized_geology_path)
            return categorized_geology_path
            
        geology_gdfs = [gpd.read_file(p) for state in state_abbrs if os.path.exists(p := os.path.join(geology_dir_path, state.upper(), f'geology_a_{state.lower()}.shp'))]
        if not geology_gdfs:
            print("Warning: No geology files found, creating NaN raster.")
            create_nan_raster_from_template(template_path, categorized_geology_path)
            return categorized_geology_path
            
        geology_gdf = gpd.GeoDataFrame(pd.concat(geology_gdfs, ignore_index=True)).to_crs(template_props['crs'])
        boundary_gdf = gpd.read_file(dissolved_boundary_path).to_crs(template_props['crs'])
        clipped_geology = gpd.clip(geology_gdf, boundary_gdf)
        
        if clipped_geology.empty:
            print("Warning: No geology data within boundary, creating NaN raster.")
            create_nan_raster_from_template(template_path, categorized_geology_path)
            return categorized_geology_path
            
        geology_field = next((f for f in ['MAJOR1', 'ROCKTYPE1', 'UNIT_NAME'] if f.upper() in [c.upper() for c in clipped_geology.columns]), None)
        if not geology_field:
            raise ValueError("Could not find a valid geology field in the shapefile.")
        
        # Save RAW clipped geology shapefile (with original text rock types)
        output_dir = os.path.dirname(categorized_geology_path)
        raw_geology_shapefile = os.path.join(output_dir, "raw_geology_clipped.shp")
        print(f"  Saving raw clipped geology shapefile to: raw_geology_clipped.shp")
        clipped_geology.to_file(raw_geology_shapefile)
        print(f"  ✓ Raw geology saved with {len(clipped_geology)} features and text rock types")
        
        # Now proceed with categorization
        geology_mapping_lower = {k.lower(): v for k, v in geology_mapping.items()}
        clipped_geology['geology_cat'] = clipped_geology[geology_field].str.lower().map(geology_mapping_lower)
        categorized_geology = clipped_geology.dropna(subset=['geology_cat'])
        
        if categorized_geology.empty:
             print("Warning: No geology features could be categorized, creating NaN raster.")
             create_nan_raster_from_template(template_path, categorized_geology_path)
             return categorized_geology_path
             
        temp_geology_shp = os.path.join(local_temp_dir, "temp_geo_cat.shp")
        categorized_geology[['geometry', 'geology_cat']].to_file(temp_geology_shp)
        
        gdal.Rasterize(categorized_geology_path, temp_geology_shp,
                       format='GTiff', outputBounds=list(template_props['bounds']),
                       width=template_props['width'], height=template_props['height'],
                       noData=np.nan, outputSRS=template_props['crs'].to_wkt(),
                       attribute='geology_cat', allTouched=True,
                       creationOptions=['COMPRESS=LZW', 'TILED=YES'])
        
        print(f"  ✓ Categorized geology raster saved: {os.path.basename(categorized_geology_path)}")
        print("✓ Geology processing completed successfully!")
        return categorized_geology_path
    except Exception as e:
        raise RuntimeError(f"Geology processing failed: {str(e)}")
    finally:
        shutil.rmtree(local_temp_dir, ignore_errors=True)


def process_and_categorize_ndvi_raster(ndvi_path: str, dem_path: str, boundary_path: str,
                                      output_path: str = None, ndvi_scale_factor: float = 10000.0,
                                      temp_dir: str = None) -> str:
    """
    Memory-efficient and robust processing of NDVI raster with categorization.
    Now saves both raw (scaled) and categorized versions.
    """
    print("Starting optimized NDVI processing and categorization...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    try:
        if output_path is None:
            basename = os.path.splitext(os.path.basename(ndvi_path))[0]
            output_path = f"{basename}_categorized.tif"
        
        # Determine raw output path
        output_dir = os.path.dirname(output_path)
        basename = os.path.splitext(os.path.basename(output_path))[0]
        # Remove '_categorized' suffix if present to create clean raw name
        if basename.endswith('_categorized'):
            basename = basename[:-len('_categorized')]
        raw_output_path = os.path.join(output_dir, "raw_scaled_ndvi_UTM.tif")
        
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_ndvi.shp")
        create_dissolved_boundary_shapefile(boundary_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem_ndvi.tif")
        template_props = create_template_from_boundary_and_dem(dissolved_boundary_path, dem_path, template_path)
        
        # Step 1: Warp the raw NDVI data to the template grid
        processed_ndvi_path = os.path.join(local_temp_dir, "ndvi_processed.tif")
        process_raster_to_template(
            ndvi_path, processed_ndvi_path, template_props, dissolved_boundary_path,
            resampling_method=Resampling.bilinear, temp_dir=local_temp_dir
        )
        
        # Step 2: Read and scale the warped data
        print("Scaling NDVI values...")
        with rasterio.open(processed_ndvi_path) as src:
            ndvi_data_int = src.read(1)
            meta = src.meta.copy()
            
            # Scale the data
            ndvi_data_float = ndvi_data_int.astype('float32') / ndvi_scale_factor
            ndvi_data_float[np.isnan(ndvi_data_int)] = np.nan # Preserve NoData
            
            # Save RAW scaled NDVI (before categorization)
            print(f"  Saving raw scaled NDVI to: {raw_output_path}")
            meta.update({'dtype': 'float32', 'nodata': np.nan})
            with rasterio.open(raw_output_path, 'w', **meta) as dst:
                dst.write(ndvi_data_float, 1)
            
            # Step 3: Categorize the scaled data
            print("Categorizing NDVI values...")
            thresholds = [-0.6, -0.2, 0.2, 0.6] # 4 thresholds for 5 categories
            categorized_data = categorize_raster_data(ndvi_data_float, 'range', thresholds=thresholds)
            
            # Save the final categorized raster
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(categorized_data, 1)

        print(f"  ✓ Raw scaled NDVI saved: {raw_output_path}")
        print(f"  ✓ Categorized NDVI saved: {output_path}")
        print(f"✓ NDVI processing completed successfully!")
        return output_path
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"NDVI processing failed: {str(e)}")
    finally:
        shutil.rmtree(local_temp_dir, ignore_errors=True)
   
# ===================================================================================================================
# ====================== Optimized Stream Buffer Raster Creation
# ====================================================================================================================
def create_stream_buffer_raster( boundary_shp_path: str, stream_vector_path: str, 
                               dem_raster_path: str, output_raster_path: str, 
                               chunk_size: int = 4096, temp_dir: str = None) -> str:
    """
    Memory-efficient creation of stream buffer raster using GDAL distance operations.
    
    This optimized version eliminates redundant clipping and chunked processing by using
    template-based processing and GDAL's efficient proximity calculations.
    
    Parameters:
    -----------

    boundary_shp_path : str
        Path to the boundary shapefile for clipping
    stream_vector_path : str
        Path to the stream vector shapefile (polylines/linestrings)
    dem_raster_path : str
        Path to the DEM raster (used as template for output raster)
    output_raster_path : str
        Path to save the output stream buffer raster
    chunk_size : int, optional
        Not used in optimized version (kept for API compatibility)
    temp_dir : str, optional
        Temporary directory for intermediate files
        
    Returns:
    --------
    str
        Path to the created stream buffer raster
    """
    print("Starting optimized stream buffer raster creation...")
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="wdfm_stream_")
        cleanup_temp = True
    else:
        cleanup_temp = False
        
    try:
        # Validate input files
        for path, name in [(boundary_shp_path, "Boundary"), (stream_vector_path, "Stream vector"), 
                          (dem_raster_path, "DEM")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
        
        # Create template from boundary and DEM
        template_path = os.path.join(temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem(
            boundary_shp_path, dem_raster_path, template_path
        )
        
        print(f"Template dimensions: {template_props['width']} x {template_props['height']}")
        print(f"Template resolution: {template_props['resolution'][0]:.2f} units per pixel")
        
        # Load and validate stream vector data
        print("Loading stream vector data...")
        try:
            stream_gdf = gpd.read_file(stream_vector_path)
            if len(stream_gdf) == 0:
                raise ValueError("Stream vector file is empty")
            
            print(f"Loaded {len(stream_gdf)} stream features")
            print(f"Stream CRS: {stream_gdf.crs}")
            print(f"Stream geometry types: {stream_gdf.geometry.type.unique()}")
            
            # Validate geometry types
            valid_geom_types = ['LineString', 'MultiLineString']
            if not all(geom_type in valid_geom_types for geom_type in stream_gdf.geometry.type.unique()):
                print("Warning: Non-line geometries found, filtering to LineString/MultiLineString only")
                stream_gdf = stream_gdf[stream_gdf.geometry.type.isin(valid_geom_types)]
            
            if len(stream_gdf) == 0:
                raise ValueError("No valid line geometries found in stream data")
                
        except Exception as e:
            raise ValueError(f"Failed to load stream vector: {str(e)}")
        
        # Transform stream data to template CRS and clip to boundary
        print("Processing stream data to template extent...")
        
        # Transform to template CRS if needed
        if stream_gdf.crs != template_props['crs']:
            print(f"Transforming streams from {stream_gdf.crs} to {template_props['crs']}")
            stream_gdf = stream_gdf.to_crs(template_props['crs'])
        
        # Clip streams to boundary for efficiency
        boundary_gdf = gpd.read_file(boundary_shp_path)
        if boundary_gdf.crs != template_props['crs']:
            boundary_gdf = boundary_gdf.to_crs(template_props['crs'])
        
        # Use spatial intersection to clip streams
        clipped_streams = gpd.clip(stream_gdf, boundary_gdf)
        
        if len(clipped_streams) == 0:
            raise ValueError("No streams intersect with the boundary area")
        
        print(f"Clipped to {len(clipped_streams)} stream features within boundary")
        
        # Save clipped streams to temporary shapefile for GDAL
        temp_streams_path = os.path.join(temp_dir, "clipped_streams.shp")
        clipped_streams.to_file(temp_streams_path)
        
        # Step 1: Rasterize streams using GDAL
        print("Rasterizing stream vectors...")
        temp_stream_raster = os.path.join(temp_dir, "stream_raster.tif")
        
        rasterize_options = gdal.RasterizeOptions(
            format='GTiff',
            outputBounds=list(template_props['bounds']),
            width=template_props['width'],
            height=template_props['height'],
            noData=0,
            outputSRS=template_props['crs'].to_wkt(),
            burnValues=[1],  # Stream pixels = 1, background = 0
            allTouched=True,
            creationOptions=['COMPRESS=LZW', 'TILED=YES']
        )
        
        gdal.Rasterize(temp_stream_raster, temp_streams_path, options=rasterize_options)
        
        # Verify stream rasterization
        with rasterio.open(temp_stream_raster) as src:
            stream_data = src.read(1)
            stream_pixels = np.sum(stream_data == 1)
            print(f"Rasterized {stream_pixels:,} stream pixels")
            
            if stream_pixels == 0:
                raise ValueError("No stream pixels created during rasterization")
        
        # Step 2: Calculate proximity (distance) using GDAL
        print("Calculating distances to streams...")
        temp_distance_raster = os.path.join(temp_dir, "distance_raster.tif")
        
        # Open stream raster and create distance raster
        stream_ds = gdal.Open(temp_stream_raster)
        driver = gdal.GetDriverByName('GTiff')
        
        # Create distance raster with same properties as stream raster
        distance_ds = driver.Create(
            temp_distance_raster, 
            template_props['width'], 
            template_props['height'], 
            1, 
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        distance_ds.SetGeoTransform(stream_ds.GetGeoTransform())
        distance_ds.SetProjection(stream_ds.GetProjection())
        distance_band = distance_ds.GetRasterBand(1)
        distance_band.SetNoDataValue(np.nan)
        
        # Calculate proximity
        gdal.ComputeProximity(
            stream_ds.GetRasterBand(1),
            distance_band,
            options=['DISTUNITS=GEO', 'VALUES=1']  # Distance to pixels with value 1 (streams)
        )
        
        # Close datasets
        stream_ds = None
        distance_ds = None
        
        print("Converting distances to vulnerability categories...")
        
        # Step 3: Categorize distances into vulnerability classes
        distance_thresholds = [30, 60, 90, 120]  # meters
        vulnerability_values = [5, 4, 3, 2, 1]   # 5=highest (closest), 1=lowest (farthest)
        
        category_names = {
            5: "0-30m from stream (highest vulnerability)",
            4: "30-60m from stream",
            3: "60-90m from stream", 
            2: "90-120m from stream",
            1: ">120m from stream (lowest vulnerability)"
        }
        
        print(f"Distance thresholds: {distance_thresholds} meters")
        
        # Read distance data and categorize
        with rasterio.open(temp_distance_raster) as dist_src:
            distance_data = dist_src.read(1)
            dist_meta = dist_src.meta.copy()
            
            # Initialize with lowest vulnerability (category 1)
            categorized_data = np.ones_like(distance_data, dtype=np.float32)
            
            # Apply distance-based categorization (work from farthest to nearest)
            for i, threshold in enumerate(reversed(distance_thresholds)):
                within_threshold = distance_data <= threshold
                vulnerability_value = vulnerability_values[len(distance_thresholds) - 1 - i]
                categorized_data[within_threshold] = vulnerability_value
            
            # Handle areas outside template (set to NaN)
            # This preserves the template boundary mask
            with rasterio.open(template_path) as template_src:
                template_data = template_src.read(1)
                template_mask = np.isnan(template_data)
                categorized_data[template_mask] = np.nan
        
        # Step 4: Save categorized buffer raster
        print("Saving categorized stream buffer raster...")
        
        output_meta = dist_meta.copy()
        output_meta.update({
            'dtype': 'float32',
            'nodata': np.nan,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_raster_path, 'w', **output_meta) as dst:
            dst.write(categorized_data, 1)
        
        # Step 5: Validate and report results
        print("Validating output...")
        
        valid_pixels = ~np.isnan(categorized_data)
        total_valid = np.sum(valid_pixels)
        
        if total_valid > 0:
            print("Category distribution:")
            for value in [5, 4, 3, 2, 1]:
                count = np.sum(categorized_data == value)
                percentage = (count / total_valid) * 100 if total_valid > 0 else 0
                print(f"  - Category {value} ({category_names[value]}): {count:,} pixels ({percentage:.1f}%)")
            
            # Verify output values
            unique_values = np.unique(categorized_data[valid_pixels])
            expected_values = {1, 2, 3, 4, 5}
            unexpected_values = set(unique_values) - expected_values
            
            if unexpected_values:
                print(f"Warning: Unexpected values found: {unexpected_values}")
            else:
                print("✓ All output values are within expected range (1-5)")
                
            print(f"✓ Total valid pixels: {total_valid:,}")
            print(f"✓ Output saved successfully: {output_raster_path}")
            
        else:
            print("Warning: No valid pixels in output raster")
        
        return output_raster_path
        
    except Exception as e:
        print(f"Error creating stream buffer raster: {e}")
        raise
        
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
# ===================================================================================================================
# ====================== Optimized Stream Buffer Raster Creation
# ====================================================================================================================
def create_stream_buffer_raster(boundary_shp_path: str, stream_vector_path: str, 
                               dem_raster_path: str, output_raster_path: str, 
                               chunk_size: int = 4096, temp_dir: str = None) -> str:
    """
    Memory-efficient creation of stream buffer raster using GDAL distance operations.
    """
    print("Starting optimized stream buffer raster creation...")
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="wdfm_stream_")
        cleanup_temp = True
    else:
        cleanup_temp = False
        
    try:
        # Create template from boundary and DEM
        template_path = os.path.join(temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem(
            boundary_shp_path, dem_raster_path, template_path
        )
        
        # Load and clip stream data
        stream_gdf = gpd.read_file(stream_vector_path)
        if stream_gdf.crs != template_props['crs']:
            stream_gdf = stream_gdf.to_crs(template_props['crs'])
        
        boundary_gdf = gpd.read_file(boundary_shp_path)
        if boundary_gdf.crs != template_props['crs']:
            boundary_gdf = boundary_gdf.to_crs(template_props['crs'])
        
        clipped_streams = gpd.clip(stream_gdf, boundary_gdf)
        if len(clipped_streams) == 0:
            raise ValueError("No streams intersect with boundary")
        
        # NEW: Filter out non-LineString geometries and explode GeometryCollections
        print(f"Original streams count: {len(clipped_streams)}")
        
        # First, explode any GeometryCollections into individual geometries
        clipped_streams = clipped_streams.explode(index_parts=False).reset_index(drop=True)
        
        # Filter to keep only LineString geometries
        valid_geom_mask = clipped_streams.geometry.geom_type == 'LineString'
        filtered_streams = clipped_streams[valid_geom_mask].copy()
        
        if len(filtered_streams) == 0:
            raise ValueError("No valid LineString geometries found after filtering")
        
        print(f"Filtered streams count: {len(filtered_streams)} (kept only LineString geometries)")
        
        # Log what was filtered out for debugging
        filtered_out = clipped_streams[~valid_geom_mask]
        if len(filtered_out) > 0:
            geom_types = filtered_out.geometry.geom_type.value_counts()
            print(f"Filtered out geometry types: {dict(geom_types)}")
        
        # Save streams for GDAL (use filtered_streams instead of clipped_streams)
        temp_streams_path = os.path.join(temp_dir, "streams.shp")
        filtered_streams.to_file(temp_streams_path)
        
        # Rest of the function remains the same...
        # Rasterize streams
        temp_stream_raster = os.path.join(temp_dir, "streams.tif")
        gdal.Rasterize(temp_stream_raster, temp_streams_path, 
                      format='GTiff', 
                      outputBounds=list(template_props['bounds']),
                      width=template_props['width'], 
                      height=template_props['height'],
                      burnValues=[1], allTouched=True)
        
        # Calculate distances
        temp_distance_raster = os.path.join(temp_dir, "distances.tif")
        stream_ds = gdal.Open(temp_stream_raster)
        distance_ds = gdal.GetDriverByName('GTiff').Create(
            temp_distance_raster, template_props['width'], template_props['height'], 1, gdal.GDT_Float32
        )
        distance_ds.SetGeoTransform(stream_ds.GetGeoTransform())
        distance_ds.SetProjection(stream_ds.GetProjection())
        
        gdal.ComputeProximity(stream_ds.GetRasterBand(1), distance_ds.GetRasterBand(1), 
                             options=['DISTUNITS=GEO', 'VALUES=1'])
        
        stream_ds = distance_ds = None
        
        # Categorize distances
        distance_thresholds = [30, 60, 90, 120]
        vulnerability_values = [5, 4, 3, 2, 1]
        
        with rasterio.open(temp_distance_raster) as src:
            distance_data = src.read(1)
            categorized_data = np.ones_like(distance_data, dtype=np.float32)
            
            for i, threshold in enumerate(reversed(distance_thresholds)):
                within_threshold = distance_data <= threshold
                vulnerability_value = vulnerability_values[len(distance_thresholds) - 1 - i]
                categorized_data[within_threshold] = vulnerability_value
            
            # Apply template mask
            with rasterio.open(template_path) as template_src:
                template_data = template_src.read(1)
                categorized_data[np.isnan(template_data)] = np.nan
            
            # Save result
            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': np.nan})
            
            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(categorized_data, 1)
        
        print(f"✓ Stream buffer raster created: {output_raster_path}")
        return output_raster_path
        
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

# ===================================================================================================================
# ====================== Optimized Road Buffer Raster Creation
# ====================================================================================================================
def create_road_buffer_raster(boundary_shp_path: str, road_vector_path: str, 
                             dem_raster_path: str, output_raster_path: str, 
                             chunk_size: int = 4096, temp_dir: str = None) -> str:
    """
    Memory-efficient creation of road buffer raster using GDAL distance operations.
    """
    print("Starting optimized road buffer raster creation...")
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="wdfm_road_")
        cleanup_temp = True
    else:
        cleanup_temp = False
        
    try:
        # Create template from boundary and DEM
        template_path = os.path.join(temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem(
            boundary_shp_path, dem_raster_path, template_path
        )
        
        # Load and clip road data
        road_gdf = gpd.read_file(road_vector_path)
        if road_gdf.crs != template_props['crs']:
            road_gdf = road_gdf.to_crs(template_props['crs'])
        
        boundary_gdf = gpd.read_file(boundary_shp_path)
        if boundary_gdf.crs != template_props['crs']:
            boundary_gdf = boundary_gdf.to_crs(template_props['crs'])
        
        clipped_roads = gpd.clip(road_gdf, boundary_gdf)
        if len(clipped_roads) == 0:
            raise ValueError("No roads intersect with boundary")
        
        # NEW: Filter out non-LineString geometries and explode GeometryCollections
        print(f"Original roads count: {len(clipped_roads)}")
        
        # First, explode any GeometryCollections into individual geometries
        clipped_roads = clipped_roads.explode(index_parts=False).reset_index(drop=True)
        
        # Filter to keep only LineString geometries
        valid_geom_mask = clipped_roads.geometry.geom_type == 'LineString'
        filtered_roads = clipped_roads[valid_geom_mask].copy()
        
        if len(filtered_roads) == 0:
            raise ValueError("No valid LineString geometries found after filtering")
        
        print(f"Filtered roads count: {len(filtered_roads)} (kept only LineString geometries)")
        
        # Log what was filtered out for debugging
        filtered_out = clipped_roads[~valid_geom_mask]
        if len(filtered_out) > 0:
            geom_types = filtered_out.geometry.geom_type.value_counts()
            print(f"Filtered out geometry types: {dict(geom_types)}")
        
        # Save roads for GDAL (use filtered_roads instead of clipped_roads)
        temp_roads_path = os.path.join(temp_dir, "roads.shp")
        filtered_roads.to_file(temp_roads_path)
        
        # Rasterize roads
        temp_road_raster = os.path.join(temp_dir, "roads.tif")
        gdal.Rasterize(temp_road_raster, temp_roads_path, 
                      format='GTiff', 
                      outputBounds=list(template_props['bounds']),
                      width=template_props['width'], 
                      height=template_props['height'],
                      burnValues=[1], allTouched=True)
        
        # Calculate distances
        temp_distance_raster = os.path.join(temp_dir, "distances.tif")
        road_ds = gdal.Open(temp_road_raster)
        distance_ds = gdal.GetDriverByName('GTiff').Create(
            temp_distance_raster, template_props['width'], template_props['height'], 1, gdal.GDT_Float32
        )
        distance_ds.SetGeoTransform(road_ds.GetGeoTransform())
        distance_ds.SetProjection(road_ds.GetProjection())
        
        gdal.ComputeProximity(road_ds.GetRasterBand(1), distance_ds.GetRasterBand(1), 
                             options=['DISTUNITS=GEO', 'VALUES=1'])
        
        road_ds = distance_ds = None
        
        # Categorize distances (same thresholds as streams)
        distance_thresholds = [30, 60, 90, 120]
        vulnerability_values = [5, 4, 3, 2, 1]
        
        with rasterio.open(temp_distance_raster) as src:
            distance_data = src.read(1)
            categorized_data = np.ones_like(distance_data, dtype=np.float32)
            
            for i, threshold in enumerate(reversed(distance_thresholds)):
                within_threshold = distance_data <= threshold
                vulnerability_value = vulnerability_values[len(distance_thresholds) - 1 - i]
                categorized_data[within_threshold] = vulnerability_value
            
            # Apply template mask
            with rasterio.open(template_path) as template_src:
                template_data = template_src.read(1)
                categorized_data[np.isnan(template_data)] = np.nan
            
            # Save result
            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': np.nan})
            
            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(categorized_data, 1)
        
        print(f"✓ Road buffer raster created: {output_raster_path}")
        return output_raster_path
        
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

# ===================================================================================================================
# ====================== Optimized WDFM Model Execution
# ====================================================================================================================
import sys
from contextlib import redirect_stdout
def run_wdfm_model(region_boundary_path, dem_raster_path,
                   wdfm_aws0150wta_path, wdfm_drainagecl_path, wdfm_kwfact_path,
                   wdfm_ksat_h_path, wdfm_runoff_path, wdfm_soilslippot_path,
                   wdfm_resdepb_r_path, wdfm_taxorder_path, wdfm_tfact_path,
                   wdfm_geology_path, wdfm_ndvi_path, wdfm_pidf_100yr_24hr_path,
                   wdfm_slope_path, wdfm_road_buffer_path, wdfm_stream_buffer_path,
                   watershed_polygon_path, output_weighted_raster_path=None,
                   output_watershed_path=None, weights_dict=None,user_id=None,
                   project_name=None,task_type = None,check_cancellation_func=None):
    """
    Memory-efficient execution of the WDFM model using template-based processing.
    """
    print("Starting optimized WDFM model execution...")
    print("=" * 60)
    
    if not all([output_weighted_raster_path, output_watershed_path, weights_dict]):
        raise ValueError("output_weighted_raster_path, output_watershed_path, and weights_dict must be provided")
    try:
        # ...
        if check_cancellation_func:
            check_cancellation_func(user_id, project_name, task_type)
        # ...
    except Exception as e:
        # Re-raise the specific TaskCancelledError if it occurs
        from app import TaskCancelledError
        if isinstance(e, TaskCancelledError):
            raise
        print(f"WDFM analysis failed: {e}")
    # Define raster info with weight mappings
    raster_info = [
        {"path": wdfm_aws0150wta_path, "weight_key": "wdfmAwswt", "name": "AWS0150WTA"},
        {"path": wdfm_drainagecl_path, "weight_key": "wdfmDrainageClasswt", "name": "DrainageClass"},
        {"path": wdfm_kwfact_path, "weight_key": "wdfmKfactwt", "name": "KFactor"},
        {"path": wdfm_ksat_h_path, "weight_key": "wdfmKsatwt", "name": "KsatHigh"},
        {"path": wdfm_runoff_path, "weight_key": "wdfmRunoffClasswt", "name": "RunoffClass"},
        {"path": wdfm_soilslippot_path, "weight_key": "wdfmSoilSlipwt", "name": "SoilSlipPotential"},
        {"path": wdfm_resdepb_r_path, "weight_key": "wdfmSoilBtmDepthwt", "name": "SoilBottomDepth"},
        {"path": wdfm_taxorder_path, "weight_key": "wdfmSoilTaxonomicwt", "name": "SoilTaxonomicOrder"},
        {"path": wdfm_tfact_path, "weight_key": "wdfmTFactorwt", "name": "TFactor"},
        {"path": wdfm_geology_path, "weight_key": "wdfmGeologyRoackType1wt", "name": "GeologyRockType1"},
        {"path": wdfm_ndvi_path, "weight_key": "wdfmNDVIwt", "name": "NDVI"},
        {"path": wdfm_pidf_100yr_24hr_path, "weight_key": "wdfmPIwt", "name": "PI"},
        {"path": wdfm_slope_path, "weight_key": "wdfmSlopewt", "name": "Slope"},
        {"path": wdfm_road_buffer_path, "weight_key": "wdfmRoadBufferwt", "name": "RoadBuffer"},
        {"path": wdfm_stream_buffer_path, "weight_key": "wdfmStreamBufferwt", "name": "StreamBuffer"}
    ]
    try:
        # ...
        if check_cancellation_func:
            check_cancellation_func(user_id, project_name, task_type)
        # ...
    except Exception as e:
        # Re-raise the specific TaskCancelledError if it occurs
        from app import TaskCancelledError
        if isinstance(e, TaskCancelledError):
            raise
        print(f"WDFM analysis failed: {e}")
    # Validate weights
    for raster in raster_info:
        if raster["weight_key"] not in weights_dict:
            raise ValueError(f"Missing weight key: {raster['weight_key']}")
        try:
            weight = float(weights_dict[raster["weight_key"]])
            if weight < 0:
                raise ValueError(f"Negative weight: {raster['weight_key']} = {weight}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid weight value: {raster['weight_key']} = {weights_dict[raster['weight_key']]}")
    try:
        # ...
        if check_cancellation_func:
            check_cancellation_func(user_id, project_name, task_type)
        # ...
    except Exception as e:
        # Re-raise the specific TaskCancelledError if it occurs
        from app import TaskCancelledError
        if isinstance(e, TaskCancelledError):
            raise
        print(f"WDFM analysis failed: {e}")
    print(f"Processing {len(raster_info)} WDFM variables with weights:")
    for raster in raster_info:
        weight = weights_dict[raster["weight_key"]]
        print(f"  - {raster['name']}: weight = {weight}")
    try:
        # ...
        if check_cancellation_func:
            check_cancellation_func(user_id, project_name, task_type)
        # ...
    except Exception as e:
        # Re-raise the specific TaskCancelledError if it occurs
        from app import TaskCancelledError
        if isinstance(e, TaskCancelledError):
            raise
        print(f"WDFM analysis failed: {e}")
    temp_dir = tempfile.mkdtemp(prefix="wdfm_model_")
    try:
        # ...
        if check_cancellation_func:
            check_cancellation_func(user_id, project_name, task_type)
        # ...
    except Exception as e:
        # Re-raise the specific TaskCancelledError if it occurs
        from app import TaskCancelledError
        if isinstance(e, TaskCancelledError):
            raise
        print(f"WDFM analysis failed: {e}")
    try:
        # Create template from region boundary and DEM
        template_path = os.path.join(temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem(
            region_boundary_path, dem_raster_path, template_path
        )
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        print(f"Template dimensions: {template_props['width']} x {template_props['height']}")
        
        # Process each raster and accumulate weighted sum
        print(f"\nProcessing {len(raster_info)} raster layers...")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Initialize weighted sum array
        weighted_sum = None
        valid_layers = 0
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        for i, raster in enumerate(raster_info, 1):
            print(f"\nProcessing {i}/{len(raster_info)}: {raster['name']}")
            weight = float(weights_dict[raster['weight_key']])
            try:
                # ...
                if check_cancellation_func:
                    check_cancellation_func(user_id, project_name, task_type)
                # ...
            except Exception as e:
                # Re-raise the specific TaskCancelledError if it occurs
                from app import TaskCancelledError
                if isinstance(e, TaskCancelledError):
                    raise
                print(f"WDFM analysis failed: {e}")
            if not os.path.exists(raster["path"]):
                print(f"  Warning: File not found - {raster['path']}")
                continue
            try:
                # ...
                if check_cancellation_func:
                    check_cancellation_func(user_id, project_name, task_type)
                # ...
            except Exception as e:
                # Re-raise the specific TaskCancelledError if it occurs
                from app import TaskCancelledError
                if isinstance(e, TaskCancelledError):
                    raise
                print(f"WDFM analysis failed: {e}")
            try:
                with rasterio.open(raster["path"]) as src:
                    # Read raster data
                    raster_data = src.read(1)
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                    # Ensure it matches template dimensions
                    if raster_data.shape != (template_props['height'], template_props['width']):
                        print(f"  Warning: Dimension mismatch for {raster['name']}")
                        print(f"    Expected: {template_props['height']}x{template_props['width']}")
                        print(f"    Got: {raster_data.shape}")
                        continue
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                    # Apply weight to valid data
                    valid_mask = ~np.isnan(raster_data)
                    valid_pixels = np.sum(valid_mask)
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                    if valid_pixels == 0:
                        print(f"  Warning: No valid data in {raster['name']}")
                        continue
                    
                    # Apply weight
                    weighted_data = np.full_like(raster_data, np.nan, dtype=np.float32)
                    weighted_data[valid_mask] = raster_data[valid_mask] * weight
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                    # Accumulate in weighted sum
                    if weighted_sum is None:
                        weighted_sum = weighted_data.copy()
                    else:
                        # Add where both have valid data
                        combined_valid = ~np.isnan(weighted_sum) | ~np.isnan(weighted_data)
                        result = np.full_like(weighted_sum, np.nan)
                        
                        # Where both are valid, add them
                        both_valid = ~np.isnan(weighted_sum) & ~np.isnan(weighted_data)
                        result[both_valid] = weighted_sum[both_valid] + weighted_data[both_valid]
                        try:
                            # ...
                            if check_cancellation_func:
                                check_cancellation_func(user_id, project_name, task_type)
                            # ...
                        except Exception as e:
                            # Re-raise the specific TaskCancelledError if it occurs
                            from app import TaskCancelledError
                            if isinstance(e, TaskCancelledError):
                                raise
                            print(f"WDFM analysis failed: {e}")
                        # Where only weighted_sum is valid, keep it
                        only_sum_valid = ~np.isnan(weighted_sum) & np.isnan(weighted_data)
                        result[only_sum_valid] = weighted_sum[only_sum_valid]
                        try:
                            # ...
                            if check_cancellation_func:
                                check_cancellation_func(user_id, project_name, task_type)
                            # ...
                        except Exception as e:
                            # Re-raise the specific TaskCancelledError if it occurs
                            from app import TaskCancelledError
                            if isinstance(e, TaskCancelledError):
                                raise
                            print(f"WDFM analysis failed: {e}")
                        # Where only weighted_data is valid, use it
                        only_data_valid = np.isnan(weighted_sum) & ~np.isnan(weighted_data)
                        result[only_data_valid] = weighted_data[only_data_valid]
                        
                        weighted_sum = result
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                    valid_layers += 1
                    print(f"  ✓ Added {raster['name']} (weight: {weight}, valid pixels: {valid_pixels:,})")
                    
                    
            except Exception as e:
                print(f"  Error processing {raster['name']}: {str(e)}")
                continue
        
        if weighted_sum is None or valid_layers == 0:
            raise ValueError("No valid raster layers were processed")
        
        print(f"\n✓ Successfully processed {valid_layers} out of {len(raster_info)} layers")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Save weighted sum raster
        print("Saving weighted sum raster...")
        with rasterio.open(template_path) as template_src:
            out_meta = template_src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': np.nan})
            
            with rasterio.open(output_weighted_raster_path, 'w', **out_meta) as dst:
                dst.write(weighted_sum.astype(np.float32), 1)
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Calculate statistics
        valid_sum_pixels = np.sum(~np.isnan(weighted_sum))
        if valid_sum_pixels > 0:
            sum_min = np.nanmin(weighted_sum)
            sum_max = np.nanmax(weighted_sum)
            sum_mean = np.nanmean(weighted_sum)
            print(f"Weighted sum statistics:")
            print(f"  Valid pixels: {valid_sum_pixels:,}")
            print(f"  Range: {sum_min:.3f} to {sum_max:.3f}")
            print(f"  Mean: {sum_mean:.3f}")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Process watersheds
        print("\nProcessing watershed polygons...")
        watersheds = gpd.read_file(watershed_polygon_path)
        
        if len(watersheds) == 0:
            raise ValueError("Watershed polygon shapefile is empty")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Transform to template CRS if needed
        if watersheds.crs != template_props['crs']:
            watersheds = watersheds.to_crs(template_props['crs'])
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Calculate sum for each watershed
        watershed_sum_values = []
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        with rasterio.open(output_weighted_raster_path) as src:
            for idx, watershed in watersheds.iterrows():
                try:
                    watershed_geom = [mapping(watershed.geometry)]
                    out_image, _ = rio_mask.mask(src, watershed_geom, crop=True, nodata=np.nan, all_touched=True)
                    valid_data = out_image[~np.isnan(out_image)]
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                    if valid_data.size > 0:
                        sum_value = np.nansum(valid_data)
                    else:
                        sum_value = np.nan
                        
                    watershed_sum_values.append(sum_value)
                    try:
                        # ...
                        if check_cancellation_func:
                            check_cancellation_func(user_id, project_name, task_type)
                        # ...
                    except Exception as e:
                        # Re-raise the specific TaskCancelledError if it occurs
                        from app import TaskCancelledError
                        if isinstance(e, TaskCancelledError):
                            raise
                        print(f"WDFM analysis failed: {e}")
                except Exception as e:
                    print(f"Error processing watershed {idx}: {str(e)}")
                    watershed_sum_values.append(np.nan)
        
        # Apply rank-based scaling to 0-5 range
        print("Applying rank-based scaling to scores...")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        valid_sums = [x for x in watershed_sum_values if not np.isnan(x)]
        if len(valid_sums) == 0:
            raise ValueError("No valid watershed sums calculated")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Rank-based scaling
        standardized_scores = np.full(len(watershed_sum_values), np.nan)
        valid_indices = [i for i, x in enumerate(watershed_sum_values) if not np.isnan(x)]
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        if len(valid_indices) > 1:
            valid_values = np.array([watershed_sum_values[i] for i in valid_indices])
            ranks = np.argsort(np.argsort(valid_values))  # Get ranks
            scaled_scores = 5.0 * ranks / (len(ranks) - 1)  # Scale to 0-5
            try:
                # ...
                if check_cancellation_func:
                    check_cancellation_func(user_id, project_name, task_type)
                # ...
            except Exception as e:
                # Re-raise the specific TaskCancelledError if it occurs
                from app import TaskCancelledError
                if isinstance(e, TaskCancelledError):
                    raise
                print(f"WDFM analysis failed: {e}")
            for i, idx in enumerate(valid_indices):
                standardized_scores[idx] = scaled_scores[i]
        elif len(valid_indices) == 1:
            standardized_scores[valid_indices[0]] = 2.5  # Middle score
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Add results to watersheds
        def score_to_category(score):
            if np.isnan(score):
                return "No Data"
            elif score < 1.0:
                return "Very Low"
            elif score < 2.0:
                return "Low"
            elif score < 3.0:
                return "Moderate"
            elif score < 4.0:
                return "High"
            else:
                return "Very High"
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Remove existing columns if present
        for col in ['wdfm_sum', 'score', 'category']:
            if col in watersheds.columns:
                watersheds = watersheds.drop(columns=[col])
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        watersheds = watersheds.copy()
        watersheds['wdfm_sum'] = watershed_sum_values
        watersheds['score'] = standardized_scores
        watersheds['category'] = [score_to_category(score) for score in standardized_scores]
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Report category distribution
        category_counts = watersheds['category'].value_counts()
        print("WDFM Category distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(watersheds)) * 100
            print(f"  {category}: {count} watersheds ({percentage:.1f}%)")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        # Save results
        watersheds.to_file(output_watershed_path)
        print(f"✓ Saved watershed results: {output_watershed_path}")
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        print("\n" + "=" * 60)
        print("WDFM Model execution completed successfully!")
        print(f"Outputs:")
        print(f"  - Weighted raster: {output_weighted_raster_path}")
        print(f"  - Watershed results: {output_watershed_path}")
        print("=" * 60)
        try:
            # ...
            if check_cancellation_func:
                check_cancellation_func(user_id, project_name, task_type)
            # ...
        except Exception as e:
            # Re-raise the specific TaskCancelledError if it occurs
            from app import TaskCancelledError
            if isinstance(e, TaskCancelledError):
                raise
            print(f"WDFM analysis failed: {e}")
        return output_watershed_path
        
    except Exception as e:
        raise ValueError(f"WDFM Model execution failed: {str(e)}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
# ===================================================================================================================
# ====================== Main WDFM Analysis Function - Memory Optimized
# ====================================================================================================================
def run_wdfm_analysis(stream_polyline_path, road_polyline_path,
                     ws_shapefile_path, original_atlas14_24hr_100yr_asci_files_dir_path,
                     dem_UTM_path, AOI_PIDF_24hr_100yr_output_raster_path,
                     AOI_PIDF_24hr_100yr_output_categorized_raster_path,
                     gssurgo_soil_data_directory_path, usa_states_shapefile_path,
                     GSSURGO_soil_data_output_dir, output_slope_percentage_AOI_raster_path,
                     categorized_output_slope_percentage_AOI_raster_path,
                     geology_dir_path, categorized_geology_path,
                     ndvi_input_path, categorized_ndvi_path,
                     categorized_stream_buffer_path, categorized_road_buffer_path,
                     wdfm_output_weighted_raster_path, wdfm_output_watershed_path,
                     wdfm_weights_dict=None, temp_dir=None, user_id=None,
                     project_name=None,
                     task_type = None,
                     check_cancellation_func=None):
    """
    Complete WDFM analysis workflow.
    
    This function orchestrates the entire WDFM (Wildfire Debris Flow Model) analysis using
    template-based processing to minimize memory usage and eliminate redundant operations.
    
    Parameters:
    -----------

    stream_polyline_path : str
        Path to stream polyline shapefile
    road_polyline_path : str
        Path to road polyline shapefile
    ws_shapefile_path : str
        Path to watershed polygon shapefile
    original_atlas14_24hr_100yr_asci_files_dir_path : str
        Directory containing NOAA Atlas 14 ASCII files
    dem_UTM_path : str
        Path to DEM raster in UTM projection
    AOI_PIDF_24hr_100yr_output_raster_path : str
        Output path for PIDF raster
    AOI_PIDF_24hr_100yr_output_categorized_raster_path : str
        Output path for categorized PIDF raster
    gssurgo_soil_data_directory_path : str
        Path to GSSURGO soil data folder
    usa_states_shapefile_path : str
        Path to US states shapefile (zipped)
    GSSURGO_soil_data_output_dir : str
        Output directory for GSSURGO processed rasters
    output_slope_percentage_AOI_raster_path : str
        Output path for slope percentage raster
    categorized_output_slope_percentage_AOI_raster_path : str
        Output path for categorized slope raster
    geology_shp_path : str
        Path to geology shapefile
    categorized_geology_path : str
        Output path for categorized geology raster
    ndvi_input_path : str
        Path to input NDVI raster
    categorized_ndvi_path : str
        Output path for categorized NDVI raster
    categorized_stream_buffer_path : str
        Output path for categorized stream buffer raster
    categorized_road_buffer_path : str
        Output path for categorized road buffer raster
    wdfm_output_weighted_raster_path : str
        Output path for final WDFM weighted raster
    wdfm_output_watershed_path : str
        Output path for watershed results with WDFM scores
    wdfm_weights_dict : Dict[str, float]
        Dictionary containing weights for all WDFM variables
    temp_dir : str, optional
        Temporary directory for intermediate files
        
    Returns:
    --------
    str
        Path to the final watershed shapefile with WDFM vulnerability scores
    """
    # Setup log file
    log_file_path = os.path.join(os.path.dirname(wdfm_output_watershed_path), 'processing_log.txt')
    
    class DualOutput:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    dual_output = DualOutput(log_file_path)
    sys.stdout = dual_output
    
    try:
        print("=" * 80)
        print("STARTING WDFM ANALYSIS WORKFLOW")
        print("=" * 80)
        
        if wdfm_weights_dict is None:
            raise ValueError("wdfm_weights_dict must be provided")
        
        # Create centralized temporary directory
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="wdfm_analysis_")
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        try:
            # Validate required input files
            required_files = [
                (stream_polyline_path, "Stream polylines"),
                (road_polyline_path, "Road polylines"),
                (ws_shapefile_path, "Watershed shapefile"),
                (dem_UTM_path, "DEM raster"),
                (ndvi_input_path, "NDVI raster"),
                (usa_states_shapefile_path, "US states shapefile")
            ]

            # Validate required directories separately
            required_dirs = [
                (geology_dir_path, "Geology data directory")
            ]

            missing_files = []
            for file_path, description in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(f"{description}: {file_path}")

            for dir_path, description in required_dirs:
                if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                    missing_files.append(f"{description}: {dir_path}")
            
            if missing_files:
                raise FileNotFoundError("Required files not found:\n" + "\n".join(missing_files))
            
            print("✓ All required input files validated")
            
            # Create output directories
            for output_path in [AOI_PIDF_24hr_100yr_output_raster_path, 
                            categorized_output_slope_percentage_AOI_raster_path,
                            categorized_geology_path, categorized_ndvi_path,
                            categorized_stream_buffer_path, categorized_road_buffer_path,
                            wdfm_output_weighted_raster_path, wdfm_output_watershed_path]:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            os.makedirs(GSSURGO_soil_data_output_dir, exist_ok=True)
            
            print("✓ Output directories created")
            
            # Step 1: Process NOAA Atlas 14 precipitation data
            print("\n" + "=" * 50)
            print("STEP 1: Processing NOAA Atlas 14 precipitation data")
            print("=" * 50)
            
            try:
                process_noaa_atlas14_for_boundary(
                    original_atlas14_24hr_100yr_asci_files_dir_path, 
                    ws_shapefile_path, 
                    dem_UTM_path, 
                    AOI_PIDF_24hr_100yr_output_raster_path,
                    AOI_PIDF_24hr_100yr_output_categorized_raster_path,
                    temp_dir=temp_dir,
                    method_prefix="wdfm"
                )
                print("✓ STEP 1 COMPLETED: Precipitation data processed")
            except Exception as e:
                raise RuntimeError(f"Step 1 failed - Precipitation processing: {str(e)}")
            
            # Step 2: Process GSSURGO soil data
            print("\n" + "=" * 50)
            print("STEP 2: Processing GSSURGO soil data")
            print("=" * 50)
            
            try:
                soil_rasters = gssurgo_to_wdfm_rasters(
                    gssurgo_soil_data_directory_path,
                    ws_shapefile_path,
                    dem_UTM_path,
                    usa_states_shapefile_path,
                    GSSURGO_soil_data_output_dir,
                    temp_dir=temp_dir
                )
                print(f"✓ STEP 2 COMPLETED: {len(soil_rasters)} soil variables processed")
            except Exception as e:
                raise RuntimeError(f"Step 2 failed - GSSURGO processing: {str(e)}")
            
            # Step 3: Calculate and categorize slope
            print("\n" + "=" * 50)
            print("STEP 3: Processing slope data")
            print("=" * 50)
            
            try:
                slope_results = calculate_and_categorize_slope(
                    
                    dem_UTM_path,
                    ws_shapefile_path, 
                    output_slope_percentage_AOI_raster_path, 
                    categorized_output_slope_percentage_AOI_raster_path,
                    temp_dir=temp_dir
                )
                
                if slope_results[0] is None:
                    raise RuntimeError("Slope calculation returned None")
                    
                print("✓ STEP 3 COMPLETED: Slope data processed")
            except Exception as e:
                raise RuntimeError(f"Step 3 failed - Slope processing: {str(e)}")
            
            # Step 4: Process geology data
            print("\n" + "=" * 50)
            print("STEP 4: Processing geology data")
            print("=" * 50)
            
            try:
                geology_result = process_geology_to_raster(
                    geology_dir_path,  
                    ws_shapefile_path, 
                    dem_UTM_path, 
                    categorized_geology_path,
                    usa_states_shapefile_path,  # Added
                    temp_dir=temp_dir
                )
                print("✓ STEP 4 COMPLETED: Geology data processed")
            except Exception as e:
                raise RuntimeError(f"Step 4 failed - Geology processing: {str(e)}")
            
            # Step 5: Process NDVI data
            print("\n" + "=" * 50)
            print("STEP 5: Processing NDVI data")
            print("=" * 50)
            
            try:
                ndvi_result = process_and_categorize_ndvi_raster(
                    
                    ndvi_input_path, 
                    dem_UTM_path, 
                    ws_shapefile_path, 
                    categorized_ndvi_path, 
                    ndvi_scale_factor=10000.0,
                    temp_dir=temp_dir
                )
                print("✓ STEP 5 COMPLETED: NDVI data processed")
            except Exception as e:
                raise RuntimeError(f"Step 5 failed - NDVI processing: {str(e)}")
            
            # Step 6: Create stream buffer raster
            print("\n" + "=" * 50)
            print("STEP 6: Creating stream buffer raster")
            print("=" * 50)
            
            try:
                stream_buffer_result = create_stream_buffer_raster(
                    
                    ws_shapefile_path, 
                    stream_polyline_path, 
                    dem_UTM_path, 
                    categorized_stream_buffer_path,
                    temp_dir=temp_dir
                )
                print("✓ STEP 6 COMPLETED: Stream buffer raster created")
            except Exception as e:
                raise RuntimeError(f"Step 6 failed - Stream buffer processing: {str(e)}")
            
            # Step 7: Create road buffer raster
            print("\n" + "=" * 50)
            print("STEP 7: Creating road buffer raster")
            print("=" * 50)
            
            try:
                road_buffer_result = create_road_buffer_raster(
                    
                    ws_shapefile_path, 
                    road_polyline_path, 
                    dem_UTM_path, 
                    categorized_road_buffer_path,
                    temp_dir=temp_dir
                )
                print("✓ STEP 7 COMPLETED: Road buffer raster created")
            except Exception as e:
                raise RuntimeError(f"Step 7 failed - Road buffer processing: {str(e)}")
            
            # Step 8: Prepare WDFM input paths
            print("\n" + "=" * 50)
            print("STEP 8: Preparing WDFM input paths")
            print("=" * 50)
            
            # Map GSSURGO outputs to WDFM variable paths
            wdfm_input_paths = {
                'wdfm_aws0150wta_path': os.path.join(GSSURGO_soil_data_output_dir, 'rootznaws_wdfm.tif'),
                'wdfm_drainagecl_path': os.path.join(GSSURGO_soil_data_output_dir, 'drainagecl_wdfm.tif'),
                'wdfm_kwfact_path': os.path.join(GSSURGO_soil_data_output_dir, 'kwfact_wdfm.tif'),
                'wdfm_ksat_h_path': os.path.join(GSSURGO_soil_data_output_dir, 'ksat_h_wdfm.tif'),  
                'wdfm_runoff_path': os.path.join(GSSURGO_soil_data_output_dir, 'runoff_wdfm.tif'),
                'wdfm_soilslippot_path': os.path.join(GSSURGO_soil_data_output_dir, 'soilslippot_wdfm.tif'),
                'wdfm_resdepb_r_path': os.path.join(GSSURGO_soil_data_output_dir, 'resdepb_r_wdfm.tif'),
                'wdfm_taxorder_path': os.path.join(GSSURGO_soil_data_output_dir, 'taxorder_wdfm.tif'),
                'wdfm_tfact_path': os.path.join(GSSURGO_soil_data_output_dir, 'tfact_wdfm.tif')
            }
            
            # Validate that required WDFM input files exist
            missing_wdfm_inputs = []
            for var_name, file_path in wdfm_input_paths.items():
                if not os.path.exists(file_path):
                    missing_wdfm_inputs.append(f"{var_name}: {file_path}")
            
            if missing_wdfm_inputs:
                print("Warning: Some GSSURGO-derived inputs are missing:")
                for missing in missing_wdfm_inputs:
                    print(f"  - {missing}")
                print("WDFM model will proceed with available inputs")
            
            print("✓ STEP 8 COMPLETED: WDFM input paths configured")
            
            # Step 9: Run WDFM model
            print("\n" + "=" * 50)
            print("STEP 9: Running WDFM model")
            print("=" * 50)
            try:
                # ...
                if check_cancellation_func:
                    check_cancellation_func(user_id, project_name, task_type)
                # ...
            except Exception as e:
                # Re-raise the specific TaskCancelledError if it occurs
                from app import TaskCancelledError
                if isinstance(e, TaskCancelledError):
                    raise
                print(f"WDFM analysis failed: {e}")
            try:
                result_path = run_wdfm_model(
                    
                    region_boundary_path=ws_shapefile_path,
                    dem_raster_path=dem_UTM_path,
                    wdfm_aws0150wta_path=wdfm_input_paths['wdfm_aws0150wta_path'],
                    wdfm_drainagecl_path=wdfm_input_paths['wdfm_drainagecl_path'],
                    wdfm_kwfact_path=wdfm_input_paths['wdfm_kwfact_path'],
                    wdfm_ksat_h_path=wdfm_input_paths['wdfm_ksat_h_path'],
                    wdfm_runoff_path=wdfm_input_paths['wdfm_runoff_path'],
                    wdfm_soilslippot_path=wdfm_input_paths['wdfm_soilslippot_path'],
                    wdfm_resdepb_r_path=wdfm_input_paths['wdfm_resdepb_r_path'],
                    wdfm_taxorder_path=wdfm_input_paths['wdfm_taxorder_path'],
                    wdfm_tfact_path=wdfm_input_paths['wdfm_tfact_path'],
                    wdfm_geology_path=categorized_geology_path,
                    wdfm_ndvi_path=categorized_ndvi_path,
                    wdfm_pidf_100yr_24hr_path=AOI_PIDF_24hr_100yr_output_categorized_raster_path,
                    wdfm_slope_path=categorized_output_slope_percentage_AOI_raster_path,
                    wdfm_road_buffer_path=categorized_road_buffer_path,
                    wdfm_stream_buffer_path=categorized_stream_buffer_path,
                    watershed_polygon_path=ws_shapefile_path,
                    output_weighted_raster_path=wdfm_output_weighted_raster_path,
                    output_watershed_path=wdfm_output_watershed_path,
                    weights_dict=wdfm_weights_dict,user_id=user_id,
                    project_name=project_name,task_type=task_type,
                    check_cancellation_func=None
                )
                print("✓ STEP 9 COMPLETED: WDFM model executed successfully")
            except Exception as e:
                raise RuntimeError(f"Step 9 failed - WDFM model execution: {str(e)}")
            
            # Final summary
            print("\n" + "=" * 80)
            print("WDFM ANALYSIS WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            print("Final Outputs:")
            print(f"  ✓ Weighted Vulnerability Raster: {wdfm_output_weighted_raster_path}")
            print(f"  ✓ Watershed Vulnerability Results: {wdfm_output_watershed_path}")
            
            print("\nIntermediate Products Created:")
            print(f"  ✓ Precipitation (PIDF): {AOI_PIDF_24hr_100yr_output_categorized_raster_path}")
            print(f"  ✓ Soil Data Directory: {GSSURGO_soil_data_output_dir}")
            print(f"  ✓ Slope: {categorized_output_slope_percentage_AOI_raster_path}")
            print(f"  ✓ Geology: {categorized_geology_path}")
            print(f"  ✓ NDVI: {categorized_ndvi_path}")
            print(f"  ✓ Stream Buffer: {categorized_stream_buffer_path}")
            print(f"  ✓ Road Buffer: {categorized_road_buffer_path}")
            
            
            print("=" * 80)
            
            return result_path
            
        except Exception as e:
            print("\n" + "=" * 80)
            print("ERROR IN WDFM ANALYSIS WORKFLOW")
            print("=" * 80)
            print(f"Error: {str(e)}")
            print("The workflow failed during processing.")
            print("Check the error message above for details.")
            print("=" * 80)
            raise
        
    finally:
        # Restore stdout and close log
        sys.stdout = dual_output.terminal
        dual_output.close()
        # Clean up centralized temporary directory
        if cleanup_temp and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"✓ Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {str(e)}")