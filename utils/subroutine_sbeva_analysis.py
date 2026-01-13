# ================================================================================================
# 📊 Optimized SBEVA Utility Functions - Memory Efficient Implementation
# (Revised for Robust Validation and CRS Handling)
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
import osmnx as ox

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
from scipy.special import softmax

# ================================================================================================
# 🌐 Web App & API Integration
# ================================================================================================
import requests

# ================================================================================================
# ⚙️ Configuration and Memory Management
# ================================================================================================
gdal.SetCacheMax(256 * 1024 * 1024)
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
gdal.SetConfigOption('GDAL_TIFF_OVR_BLOCKSIZE', '512')

wbt = whitebox.WhiteboxTools()

# ================================================================================================
# 🔧 Core Helper Functions for SBEVA
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

def create_template_from_boundary_and_dem_sbeva(boundary_path: str, dem_path: str,
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

def process_raster_to_template_sbeva(input_path: str, output_path: str, template_props: Dict[str, Any],
                                    boundary_path: str, resampling_method: Resampling = Resampling.bilinear,
                                    temp_dir: str = None, input_crs_override: str = None) -> str:
    """Robustly processes any raster to match template properties using a safe, multi-step approach."""
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

def cleanup_shapefile_sbeva(shapefile_path: str):
    """Clean up all components of a shapefile."""
    base_path = os.path.splitext(shapefile_path)[0]
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml']:
        file_path = base_path + ext
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

def apply_softmax_categorization_sbeva(data: np.ndarray, n_categories: int = 5) -> np.ndarray:
    """Apply softmax-based categorization to data for SBEVA processing."""
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full_like(data, np.nan, dtype=np.float32)
    valid_data = data[valid_mask]
    unique_values = np.sort(np.unique(valid_data))
    if len(unique_values) <= 1:
        result = np.full_like(data, np.nan, dtype=np.float32)
        result[valid_mask] = (n_categories + 1) // 2
        return result
    normalized_values = (unique_values - np.min(unique_values)) / (np.max(unique_values) - np.min(unique_values) + 1e-10)
    sm_values = softmax(normalized_values)
    cum_sm = np.cumsum(sm_values)
    thresholds = []
    for i in range(1, n_categories):
        target = i / n_categories
        idx = np.searchsorted(cum_sm, target)
        if idx < len(unique_values):
            thresholds.append(unique_values[idx])
        else:
            thresholds.append(np.max(unique_values))
    result = np.full_like(data, np.nan, dtype=np.float32)
    result[valid_mask & (data <= thresholds[0])] = 1
    for i in range(1, len(thresholds)):
        result[valid_mask & (data > thresholds[i-1]) & (data <= thresholds[i])] = i + 1
    result[valid_mask & (data > thresholds[-1])] = n_categories
    return result

def apply_linear_scaling_sbeva(data: np.ndarray, min_val: float = 1.0, max_val: float = 5.0) -> np.ndarray:
    """Apply linear scaling to data for SBEVA processing."""
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full_like(data, np.nan, dtype=np.float32)
    valid_data = data[valid_mask]
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    if data_max == data_min:
        result = np.full_like(data, np.nan, dtype=np.float32)
        result[valid_mask] = (min_val + max_val) / 2
        return result
    result = np.full_like(data, np.nan, dtype=np.float32)
    result[valid_mask] = min_val + (valid_data - data_min) / (data_max - data_min) * (max_val - min_val)
    return result

def get_us_states_crossed_sbeva(polygon_path: str, usa_states_shapefile_path: str,
                                state_abbr_column: str = "stusps") -> List[str]:
    """Finds the US state abbreviations that the given boundary polygon crosses for SBEVA."""
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

def create_nan_raster_from_template_sbeva(template_path: str, output_path: str):
    """Helper function to create a NaN raster matching template specifications for SBEVA."""
    with rasterio.open(template_path) as template:
        nan_data = np.full((template.height, template.width), np.nan, dtype=np.float32)
        out_meta = template.meta.copy()
        out_meta.update({'dtype': 'float32', 'nodata': np.nan})
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(nan_data, 1)

def apply_water_mask_sbeva(data: np.ndarray, water_class: int = 11) -> np.ndarray:
    """Convert water pixels to NaN for SBEVA processing."""
    result = data.copy().astype(np.float32)
    result[data == water_class] = np.nan
    return result

def rank_based_scale_sbeva(values: np.ndarray, min_score: float = 0.0, max_score: float = 5.0) -> np.ndarray:
    """Apply rank-based scaling for SBEVA vulnerability scoring."""
    values_array = np.array(values)
    valid_mask = ~np.isnan(values_array)
    if not np.any(valid_mask):
        return values_array
    if np.all(values_array[valid_mask] == values_array[valid_mask][0]):
        result = np.full_like(values_array, (min_score + max_score) / 2)
        result[~valid_mask] = np.nan
        return result
    valid_values = values_array[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    rank_order = np.argsort(valid_values)
    n_valid = len(valid_values)
    if n_valid > 1:
        ranks = np.linspace(min_score, max_score, n_valid)
        scaled_values = np.zeros_like(valid_values)
        scaled_values[rank_order] = ranks
    else:
        scaled_values = np.array([(min_score + max_score) / 2])
    result = np.full_like(values_array, np.nan)
    result[valid_indices] = scaled_values
    return result

def score_to_category_sbeva(score: float) -> str:
    """Convert SBEVA vulnerability scores to categories."""
    if np.isnan(score): return "No Data"
    elif score < 1.0: return "Very Low"
    elif score < 2.0: return "Low"
    elif score < 3.0: return "Moderate"
    elif score < 4.0: return "High"
    else: return "Very High"

def create_stream_buffer_geometry_sbeva(stream_path: str, boundary_path: str,
                                       buffer_distance: float, temp_dir: str = None) -> List[Dict]:
    """Create buffered stream geometry for SBEVA processing."""
    print(f"Creating stream buffer with {buffer_distance}m distance...")
    streams = gpd.read_file(stream_path)
    boundary = gpd.read_file(boundary_path)
    if streams.crs != boundary.crs:
        streams = streams.to_crs(boundary.crs)
    clipped_streams = gpd.clip(streams, boundary)
    if len(clipped_streams) == 0:
        raise ValueError("No streams intersect with boundary")
    stream_buffer = clipped_streams.buffer(buffer_distance)
    buffer_geoms = [mapping(geom) for geom in stream_buffer]
    print(f"Created buffer for {len(clipped_streams)} stream features")
    return buffer_geoms

# FIX 1: The validation function is rewritten to be robust.
def validate_raster_data_sbeva(raster_path: str, raster_name: str) -> bool:
    """
    Robustly validates that a raster contains ANY valid data, not just in the top-left corner.
    """
    if not os.path.exists(raster_path):
        print(f"Warning: {raster_name} not found at {raster_path}")
        return False
    try:
        with rasterio.open(raster_path) as src:
            # Read the entire raster's data. For validation, this is the safest approach.
            # Using a masked array is memory-efficient for checking.
            data = src.read(1, masked=True)
            # Check if there are any unmasked pixels (i.e., valid data) anywhere in the raster.
            if np.any(data.mask == False):
                return True
            else:
                print(f"Warning: {raster_name} contains only NaN values")
                return False
    except Exception as e:
        print(f"Error validating {raster_name}: {str(e)}")
        return False

def create_erosion_mapping_sbeva(erosion_csv_path: str) -> Dict[int, int]:
    """Create mapping from NLCD classes to erosion susceptibility codes for SBEVA."""
    erosion_df = pd.read_csv(erosion_csv_path)
    erosion_scale = {"Very Low": 1, "Low": 2, "Low-Medium": 3, "Medium": 4, "Medium-High": 5, "High": 6, "Very High": 7}
    erosion_mapping = {}
    for _, row in erosion_df.iterrows():
        class_value = row['Class Value']
        susceptibility = row['General Soil Erosion Susceptibility']
        erosion_code = erosion_scale.get(susceptibility, 0)
        erosion_mapping[class_value] = erosion_code
    return erosion_mapping

# ===================================================================================================================
# ====================== Optimized Data Processing Workflows ========================================================
# ===================================================================================================================

def process_noaa_atlas14_for_boundary(atlas14_dir: str, boundary_shapefile: str, dem_path: str,
                                      output_raster_path: str, categorized_raster_path: str = None,
                                      temp_dir: str = None, method_prefix: str = "sbeva") -> str:
    """Processes NOAA Atlas 14 data robustly using a standardized, template-based approach."""
    print("Processing NOAA Atlas 14 with resolution-adaptive approach...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    cleanup_temp = temp_dir is None
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary.shp")
        create_dissolved_boundary_shapefile(boundary_shapefile, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem_sbeva(
            dissolved_boundary_path, dem_path, template_path
        )
        print("Template created:")
        print(f"  CRS: {template_props['crs']}")
        print(f"  Resolution: {template_props['resolution'][0]:.2f}m")
        print(f"  Dimensions: {template_props['width']} x {template_props['height']}")
        asc_files = glob.glob(os.path.join(atlas14_dir, "**", "*.asc"), recursive=True)
        if not asc_files:
            raise ValueError("No ASC files found in the specified directory")
        print(f"Found {len(asc_files)} ASC files")
        boundary_gdf = gpd.read_file(dissolved_boundary_path).to_crs("EPSG:4326")
        boundary_for_intersect = boundary_gdf.unary_union
        intersecting_files = []
        for asc_file in asc_files:
            try:
                with rasterio.open(asc_file) as src:
                    src_bounds_wgs84 = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    if boundary_for_intersect.intersects(box(*src_bounds_wgs84)):
                        intersecting_files.append(asc_file)
            except Exception as e:
                print(f"Warning: Could not check {os.path.basename(asc_file)}: {e}")
        if not intersecting_files:
            raise ValueError("No ASC files were found that intersect with the boundary area.")
        print(f"Processing {len(intersecting_files)} intersecting files")
        processed_rasters = []
        for i, asc_file in enumerate(intersecting_files):
            temp_processed_path = os.path.join(local_temp_dir, f"processed_{i}.tif")
            try:
                process_raster_to_template_sbeva(
                    asc_file, temp_processed_path, template_props, dissolved_boundary_path,
                    resampling_method=Resampling.bilinear, temp_dir=local_temp_dir
                )
                if validate_raster_data_sbeva(temp_processed_path, os.path.basename(asc_file)):
                     processed_rasters.append(temp_processed_path)
                     print(f"  ✓ SUCCESS: {os.path.basename(asc_file)} processed with valid data.")
                else:
                     print(f"  ✗ FAILED: {os.path.basename(asc_file)} validation check failed (empty raster).")
            except Exception as e:
                print(f"  ✗ ERROR processing {os.path.basename(asc_file)}: {e}")
                traceback.print_exc()
        if not processed_rasters:
            raise ValueError("No valid processed rasters could be generated after processing intersecting files.")
        if len(processed_rasters) > 1:
            print(f"Merging {len(processed_rasters)} processed rasters...")
            final_raster_path = os.path.join(local_temp_dir, "merged.tif")
            vrt_path = os.path.join(local_temp_dir, "merged.vrt")
            gdal.BuildVRT(vrt_path, processed_rasters)
            gdal.Translate(final_raster_path, vrt_path)
        else:
            final_raster_path = processed_rasters[0]
        shutil.copy2(final_raster_path, output_raster_path)
        print(f"✓ Saved final raster: {output_raster_path}")
        if categorized_raster_path:
            print("Creating categorized version using softmax...")
            with rasterio.open(output_raster_path) as src:
                data = src.read(1)
                categorized_data = apply_softmax_categorization_sbeva(data, n_categories=5)
                cat_meta = src.meta.copy()
                with rasterio.open(categorized_raster_path, 'w', **cat_meta) as dst:
                    dst.write(categorized_data, 1)
            print(f"✓ Saved categorized raster: {categorized_raster_path}")
        return output_raster_path
    except Exception as e:
        raise RuntimeError(f"NOAA Atlas 14 processing failed: {str(e)}")
    finally:
        if cleanup_temp and os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)

def process_prism_raster(input_raster_path: str, dem_raster_path: str, boundary_shp_path: str,
                        output_raster_path: str, categorized_output_path: str, temp_dir: str,
                        var: str = 'precip') -> Tuple[str, str]:
    print(f"Starting optimized PRISM {var} processing...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    cleanup_temp = temp_dir is None
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, f"dissolved_boundary_{var}.shp")
        create_dissolved_boundary_shapefile(boundary_shp_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, f"template_dem_{var}.tif")
        template_props = create_template_from_boundary_and_dem_sbeva(dissolved_boundary_path, dem_raster_path, template_path)
        process_raster_to_template_sbeva(
            input_raster_path, output_raster_path, template_props, dissolved_boundary_path,
            resampling_method=Resampling.bilinear, temp_dir=local_temp_dir, input_crs_override='EPSG:4269'
        )
        print(f"✓ Saved processed {var} raster: {output_raster_path}")
        with rasterio.open(output_raster_path) as src:
            data = src.read(1)
            categorized_data = apply_linear_scaling_sbeva(data, min_val=1.0, max_val=5.0)
            cat_meta = src.meta.copy()
            cat_meta.update({'dtype': 'float32', 'nodata': np.nan})
            with rasterio.open(categorized_output_path, 'w', **cat_meta) as dst:
                dst.write(categorized_data.astype(np.float32), 1)
        print(f"✓ Saved categorized {var} raster: {categorized_output_path}")
        return output_raster_path, categorized_output_path
    except Exception as e:
        raise RuntimeError(f"PRISM {var} processing failed: {str(e)}")
    finally:
        if cleanup_temp:
            shutil.rmtree(local_temp_dir, ignore_errors=True)

def gssurgo_to_variable_rasters(base_raster_path: str, boundary_shp_path: str,
                                dem_path: str, output_dir: str, variables_list: List[str],
                                usa_states_shapefile_path: str, temp_dir: str = None,
                                state_abbr_column: str = "stusps") -> List[str]:
    print("Starting optimized GSSURGO processing for SBEVA...")
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    cleanup_temp = temp_dir is None
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_gssurgo.shp")
        create_dissolved_boundary_shapefile(boundary_shp_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem_sbeva(dissolved_boundary_path, dem_path, template_path)
        state_abbr_list = get_us_states_crossed_sbeva(dissolved_boundary_path, usa_states_shapefile_path, state_abbr_column)
        if not state_abbr_list:
            print("No states found that intersect with the boundary polygon.")
            return []
        print(f"Found {len(state_abbr_list)} states: {state_abbr_list}")
        
        for variable in variables_list:
            print(f"\nProcessing variable: {variable}")
            try:
                source_raster_paths = [p for state_abbr in state_abbr_list if os.path.exists(p := os.path.join(base_raster_path, state_abbr, f"{variable}.tif"))]
                
                # Define output paths - raw and processed versions
                raw_output_path = os.path.join(output_dir, f"raw_{variable}_raster.tif")
                output_raster_path = os.path.join(output_dir, f"{variable}_raster.tif")

                if not source_raster_paths:
                    print(f"  No source rasters found for '{variable}', creating NaN raster")
                    create_nan_raster_from_template_sbeva(template_path, raw_output_path)
                    # Copy to non-raw version for consistency
                    shutil.copy2(raw_output_path, output_raster_path)
                    output_paths.append(output_raster_path)
                    continue
                
                input_for_processing = source_raster_paths[0]
                if len(source_raster_paths) > 1:
                    vrt_path = os.path.join(local_temp_dir, f"{variable}_merge.vrt")
                    gdal.BuildVRT(vrt_path, source_raster_paths)
                    input_for_processing = vrt_path
                
                # Process the raw data and save it with 'raw_' prefix
                print(f"  Saving raw clipped data to: raw_{variable}_raster.tif")
                process_raster_to_template_sbeva(
                    input_for_processing, raw_output_path, template_props, dissolved_boundary_path,
                    resampling_method=Resampling.nearest, temp_dir=local_temp_dir, input_crs_override='EPSG:5070'
                )
                
                # For rootznaws ONLY, create a categorized version
                if variable == 'rootznaws':
                    print("  Applying softmax categorization and inverting scale for 'rootznaws'...")
                    
                    # Read the raw data
                    with rasterio.open(raw_output_path, 'r') as src:
                        meta = src.meta.copy()
                        data = src.read(1)

                    # 1. Scale data to 1-5 using softmax
                    categorized_data = apply_softmax_categorization_sbeva(data, n_categories=5)
                    
                    # 2. Invert the scale (1->5, 2->4, ..., 5->1)
                    valid_mask = ~np.isnan(categorized_data)
                    categorized_data[valid_mask] = 6 - categorized_data[valid_mask]
                    
                    # 3. Update metadata for the new float data
                    meta.update(dtype='float32', nodata=np.nan)
                    
                    # 4. Save the categorized version (without 'raw_' prefix)
                    with rasterio.open(output_raster_path, 'w', **meta) as dst:
                        dst.write(categorized_data.astype(np.float32), 1)
                    
                    print(f"  ✓ Raw data saved: {raw_output_path}")
                    print(f"  ✓ Categorized data saved: {output_raster_path}")
                else:
                    # For other variables (runoff, drainagecl, hydgrp), 
                    # they are already categorical codes - just copy raw to regular output
                    print(f"  Variable '{variable}' contains categorical codes (not categorizing)")
                    shutil.copy2(raw_output_path, output_raster_path)
                    print(f"  ✓ Raw data saved: {raw_output_path}")
                    print(f"  ✓ Output data saved: {output_raster_path}")
              
                print(f"  ✓ Success: {variable} processing complete")
                output_paths.append(output_raster_path)
            except Exception as e:
                print(f"  Error processing '{variable}': {str(e)}")
        
        print(f"\n✓ GSSURGO processing complete: {len(output_paths)} rasters created")
        return output_paths
    except Exception as e:
        raise RuntimeError(f"GSSURGO processing failed: {str(e)}")
    finally:
        if cleanup_temp:
            shutil.rmtree(local_temp_dir, ignore_errors=True)

def calculate_and_categorize_slope(dem_path: str, boundary_shapefile_path: str,
                                  slope_percentage_path: str, categorized_slope_path: str,
                                  temp_dir: str = None) -> Tuple[str, str, List[float]]:
    print("Starting optimized slope calculation for SBEVA...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    cleanup_temp = temp_dir is None
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_slope.shp")
        create_dissolved_boundary_shapefile(boundary_shapefile_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem.tif")
        create_template_from_boundary_and_dem_sbeva(dissolved_boundary_path, dem_path, template_path)
        print("Calculating slope using GDAL...")
        slope_options = gdal.DEMProcessingOptions(
            format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES'], slopeFormat='percent',
            computeEdges=True, alg='ZevenbergenThorne'
        )
        gdal.DEMProcessing(slope_percentage_path, template_path, 'slope', options=slope_options)
        print("Categorizing slope using softmax...")
        with rasterio.open(slope_percentage_path) as slope_src:
            slope_data = slope_src.read(1)
            slope_meta = slope_src.meta.copy()
            slope_data[slope_data < 0] = np.nan
            categorized_data = apply_softmax_categorization_sbeva(slope_data, n_categories=5)
            slope_meta.update({'dtype': 'float32', 'nodata': np.nan})
            with rasterio.open(categorized_slope_path, 'w', **slope_meta) as dst:
                dst.write(categorized_data.astype(np.float32), 1)
        print("✓ Slope calculation and categorization completed successfully!")
        return slope_percentage_path, categorized_slope_path, []
    except Exception as e:
        raise RuntimeError(f"Slope calculation failed: {str(e)}")
    finally:
        if cleanup_temp:
            shutil.rmtree(local_temp_dir, ignore_errors=True)

def process_nlcd_with_erosion(ws_shapefile_path: str, dem_path: str, nlcd_2021_conus_raster_path: str,
                             erosion_csv_path: str, output_nlcd_raster_path: str,
                             output_erosion_raster_path: str, temp_dir: str = None) -> Tuple[str, str]:
    print("Starting optimized NLCD with erosion processing for SBEVA...")
    local_temp_dir = tempfile.mkdtemp(dir=temp_dir)
    cleanup_temp = temp_dir is None
    try:
        dissolved_boundary_path = os.path.join(local_temp_dir, "dissolved_boundary_nlcd.shp")
        create_dissolved_boundary_shapefile(ws_shapefile_path, dissolved_boundary_path)
        template_path = os.path.join(local_temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem_sbeva(dissolved_boundary_path, dem_path, template_path)
        print("Processing NLCD (.img format) to template...")
        process_raster_to_template_sbeva(
            nlcd_2021_conus_raster_path, output_nlcd_raster_path, template_props, dissolved_boundary_path,
            resampling_method=Resampling.nearest, temp_dir=local_temp_dir, input_crs_override='EPSG:5070'
        )
        print(f"✓ Saved processed NLCD raster: {output_nlcd_raster_path}")
        print("Creating erosion susceptibility raster...")
        erosion_mapping = create_erosion_mapping_sbeva(erosion_csv_path)
        with rasterio.open(output_nlcd_raster_path) as src:
            nlcd_data = src.read(1)
            erosion_data = np.full_like(nlcd_data, np.nan, dtype=np.float32)
            for nlcd_value, erosion_code in erosion_mapping.items():
                erosion_data[nlcd_data == nlcd_value] = erosion_code
            erosion_meta = src.meta.copy()
            erosion_meta.update({'dtype': 'float32', 'nodata': np.nan})
            with rasterio.open(output_erosion_raster_path, 'w', **erosion_meta) as dst:
                dst.write(erosion_data, 1)
        print(f"✓ Saved erosion susceptibility raster: {output_erosion_raster_path}")
        return output_nlcd_raster_path, output_erosion_raster_path
    except Exception as e:
        raise RuntimeError(f"NLCD processing failed: {str(e)}")
    finally:
        if cleanup_temp:
            shutil.rmtree(local_temp_dir, ignore_errors=True)


# ===================================================================================================================
# ====================== Optimized SBEVA Model Execution
# ====================================================================================================================
def run_sbeva_model(dem_raster_path, sbeva24hr100yrPI_path,
                   sbeva30yrPrecipNorm_path, sbeva30yrTempNorm_path,
                   sbeva30yrSolarNorm_path, sbevaAWSrtzn_path,
                   sbevaRunoffClass_path, sbevaDrainageClass_path,
                   sbevaHydroSoilGrp_path, sbevaSlope_path,
                   sbevaLandCover_path, stream_polyline_path,
                   watershed_polygon_path, output_weighted_raster_path = None,
                   output_watershed_path = None, output_stream_buffer_path = None,
                   weights_dict= None, stream_buffer_distance = 20,
                   chunk_size = 4096, temp_dir = None,user_id=None,
                   project_name=None,task_type=None,
                   check_cancellation_func=None): 
    """
    Memory-efficient execution of the SBEVA model using template-based processing.
    
    This optimized version eliminates redundant clipping operations by using a consistent
    template and processes stream buffer areas efficiently.
    
    Parameters:
    -----------
    [Previous parameter documentation remains the same]
    
    Returns:
    --------
    None
        Files are written to disk
    """
    print("Starting optimized SBEVA model execution...")
    
    # Validate required parameters
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
        print(f"SBEVA analysis failed: {e}")
    # Define raster info with weight mappings
    raster_info = [
        {"path": sbeva24hr100yrPI_path, "weight_key": "sbeva24hr100yrPIwt", "name": "24hr100yrPI"},
        {"path": sbeva30yrPrecipNorm_path, "weight_key": "sbeva30yrPrecipNormwt", "name": "30yrPrecipNorm"},
        {"path": sbeva30yrTempNorm_path, "weight_key": "sbeva30yrTempNormwt", "name": "30yrTempNorm"},
        {"path": sbeva30yrSolarNorm_path, "weight_key": "sbeva30yrSolarNormwt", "name": "30yrSolarNorm"},
        {"path": sbevaAWSrtzn_path, "weight_key": "sbevaAWSrtznwt", "name": "AWSrtzn"},
        {"path": sbevaRunoffClass_path, "weight_key": "sbevaRunoffClasswt", "name": "RunoffClass"},
        {"path": sbevaDrainageClass_path, "weight_key": "sbevaDrainageClasswt", "name": "DrainageClass"},
        {"path": sbevaHydroSoilGrp_path, "weight_key": "sbevaHydroSoilGrpwt", "name": "HydroSoilGrp"},
        {"path": sbevaSlope_path, "weight_key": "sbevaSlopewt", "name": "Slope"},
        {"path": sbevaLandCover_path, "weight_key": "sbevaLandCoverwt", "name": "LandCover"}
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
        print(f"SBEVA analysis failed: {e}")
    # Validate weights
    for raster in raster_info:
        if raster["weight_key"] not in weights_dict:
            raise ValueError(f"Weight key {raster['weight_key']} not found in weights dictionary")
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="sbeva_model_")
        cleanup_temp = True
    else:
        cleanup_temp = False
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
        print(f"SBEVA analysis failed: {e}")
    try:
        print("Creating stream buffer area...")
        
        # Create stream buffer geometry efficiently
        stream_buffer_geoms = create_stream_buffer_geometry_sbeva(
            stream_polyline_path, watershed_polygon_path, stream_buffer_distance, temp_dir
        )
        
        # Save stream buffer if requested
        if output_stream_buffer_path:
            streams = gpd.read_file(stream_polyline_path)
            boundary = gpd.read_file(watershed_polygon_path)
            if streams.crs != boundary.crs:
                streams = streams.to_crs(boundary.crs)
            
            clipped_streams = gpd.clip(streams, boundary)
            stream_buffer_gdf = gpd.GeoDataFrame(
                geometry=clipped_streams.buffer(stream_buffer_distance), 
                crs=clipped_streams.crs
            )
            stream_buffer_gdf['buffer_dis'] = stream_buffer_distance
            stream_buffer_gdf.to_file(output_stream_buffer_path)
            print(f"✓ Saved stream buffer: {output_stream_buffer_path}")
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
            print(f"SBEVA analysis failed: {e}")
        # Create template from watershed and DEM, then clip to stream buffer
        print("Creating template clipped to stream buffer area...")
        template_path = os.path.join(temp_dir, "template_dem.tif")
        template_props = create_template_from_boundary_and_dem_sbeva(watershed_polygon_path, dem_raster_path, template_path)
        
        # Clip template to stream buffer area
        with rasterio.open(template_path) as template_src:
            stream_buffer_data, stream_buffer_transform = rio_mask.mask(
                template_src, stream_buffer_geoms, crop=True, nodata=np.nan, all_touched=True
            )
            
            # Update template properties for stream buffer area
            template_props.update({
                'transform': stream_buffer_transform,
                'width': stream_buffer_data.shape[2],
                'height': stream_buffer_data.shape[1],
                'bounds': rasterio.transform.array_bounds(
                    stream_buffer_data.shape[1], stream_buffer_data.shape[2], stream_buffer_transform
                )
            })
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
            print(f"SBEVA analysis failed: {e}")
        print(f"Stream buffer template: {template_props['width']} x {template_props['height']}")
        
        # Process each raster to stream buffer template and apply weights
        print("Processing SBEVA variables to stream buffer area...")
        
        weighted_sum = None
        valid_rasters = 0
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
            print(f"SBEVA analysis failed: {e}")
        for raster in raster_info:
            print(f"\nProcessing {raster['name']}")
            
            # Validate raster exists and has data
            if not validate_raster_data_sbeva(raster["path"], raster["name"]):
                print(f"  Skipping {raster['name']} - invalid or missing data")
                continue
            
            weight = float(weights_dict[raster["weight_key"]])
            print(f"  Weight: {weight}")
            
            try:
                # Read and process raster to match template
                with rasterio.open(raster["path"]) as src:
                    # Read full raster data
                    src_data = src.read(1)
                    
                    # Check if reprojection/resampling is needed
                    if (src.height != template_props['height'] or 
                        src.width != template_props['width'] or
                        src.transform != template_props['transform']):
                        
                        print(f"  Resampling to template dimensions")
                        
                        # Create output array matching template
                        final_data = np.full((template_props['height'], template_props['width']), 
                                            np.nan, dtype=np.float32)
                        
                        reproject(
                            source=src_data.astype(np.float32),
                            destination=final_data,
                            src_transform=src.transform,  # Use original raster transform
                            dst_transform=template_props['transform'],
                            src_crs=src.crs,
                            dst_crs=template_props['crs'],
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan
                        )
                    else:
                        # Dimensions and transform match perfectly
                        final_data = src_data.astype(np.float32)
                    
                    # Create a rasterized buffer mask at template resolution
                    from rasterio.features import rasterize
                    
                    # Convert buffer geometries to shapes
                    buffer_shapes = [(geom, 1) for geom in stream_buffer_geoms]
                    
                    # Rasterize buffer at template dimensions
                    buffer_mask = rasterize(
                        buffer_shapes,
                        out_shape=(template_props['height'], template_props['width']),
                        transform=template_props['transform'],
                        fill=0,
                        dtype=np.uint8
                    )
                    
                    # Apply mask: keep only data within buffer (where mask == 1)
                    final_data[buffer_mask == 0] = np.nan
                
                # Apply weight
                valid_mask = ~np.isnan(final_data)
                if not np.any(valid_mask):
                    print(f"  Warning: No valid data after processing")
                    continue
                
                weighted_data = np.full_like(final_data, np.nan)
                weighted_data[valid_mask] = final_data[valid_mask] * weight
                
                # Accumulate in weighted sum
                if weighted_sum is None:
                    weighted_sum = weighted_data.copy()
                else:
                    # Add where both have valid data, preserve where only one has data
                    both_valid = ~np.isnan(weighted_sum) & ~np.isnan(weighted_data)
                    only_sum_valid = ~np.isnan(weighted_sum) & np.isnan(weighted_data)
                    only_data_valid = np.isnan(weighted_sum) & ~np.isnan(weighted_data)
                    
                    result = np.full_like(weighted_sum, np.nan)
                    result[both_valid] = weighted_sum[both_valid] + weighted_data[both_valid]
                    result[only_sum_valid] = weighted_sum[only_sum_valid]
                    result[only_data_valid] = weighted_data[only_data_valid]
                    
                    weighted_sum = result
                
                valid_rasters += 1
                valid_pixels = np.sum(valid_mask)
                print(f"  ✓ Added {raster['name']} (valid pixels: {valid_pixels:,})")
                
                
            except Exception as e:
                print(f"  Error processing {raster['name']}: {str(e)}")
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
            print(f"SBEVA analysis failed: {e}")
        if weighted_sum is None or valid_rasters == 0:
            raise ValueError("No valid raster layers were processed")
        
        print(f"\n✓ Successfully processed {valid_rasters} out of {len(raster_info)} layers")
        
        # Save weighted sum raster
        print("Saving weighted sum raster...")
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
            print(f"SBEVA analysis failed: {e}")
        output_meta = {
            'driver': 'GTiff',
            'height': template_props['height'],
            'width': template_props['width'],
            'count': 1,
            'dtype': 'float32',
            'crs': template_props['crs'],
            'transform': template_props['transform'],
            'nodata': np.nan,
            'compress': 'lzw'
        }
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
            print(f"SBEVA analysis failed: {e}")
        with rasterio.open(output_weighted_raster_path, 'w', **output_meta) as dst:
            dst.write(weighted_sum.astype(np.float32), 1)
        
        # Calculate and report statistics
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
            print(f"SBEVA analysis failed: {e}")
        # Process watersheds
        print("Processing watershed polygons...")
        watersheds = gpd.read_file(watershed_polygon_path)
        
        # Handle duplications
        if 'Point_ID' in watersheds.columns:
            duplicate_count = watersheds['Point_ID'].duplicated().sum()
            if duplicate_count > 0:
                print(f"Found {duplicate_count} duplicated Point_ID values, dissolving...")
                original_count = len(watersheds)
                watersheds = watersheds.dissolve(by='Point_ID', as_index=False)
                print(f"After dissolving: {len(watersheds)} unique watersheds (reduced from {original_count})")
        
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
            print(f"SBEVA analysis failed: {e}")
        # Calculate sum for each watershed
        watershed_sum_values = []
        
        with rasterio.open(output_weighted_raster_path) as src:
            for idx, watershed in watersheds.iterrows():
                try:
                    watershed_geom = [mapping(watershed.geometry)]
                    out_image, _ = rio_mask.mask(src, watershed_geom, crop=True, nodata=np.nan, all_touched=True)
                    valid_data = out_image[~np.isnan(out_image)]
                    
                    if valid_data.size > 0:
                        sum_value = np.nansum(valid_data)
                    else:
                        sum_value = np.nan
                        
                    watershed_sum_values.append(sum_value)
                    
                except Exception as e:
                    print(f"Error processing watershed {idx}: {str(e)}")
                    watershed_sum_values.append(np.nan)
        
        # Apply rank-based scaling and categorization
        print("Applying rank-based scaling and categorization...")
        standardized_scores = rank_based_scale_sbeva(watershed_sum_values, min_score=0.0, max_score=5.0)
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
            print(f"SBEVA analysis failed: {e}")
        # Add results to watersheds
        for col in ['sum', 'score', 'category']:
            if col in watersheds.columns:
                watersheds = watersheds.drop(columns=[col])
        
        watersheds = watersheds.copy()
        watersheds['sum'] = watershed_sum_values
        watersheds['score'] = standardized_scores
        watersheds['category'] = [score_to_category_sbeva(score) for score in standardized_scores]
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
            print(f"SBEVA analysis failed: {e}")
        # Report category distribution
        category_counts = watersheds['category'].value_counts()
        print("SBEVA Category distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(watersheds)) * 100
            print(f"  {category}: {count} watersheds ({percentage:.1f}%)")
        
        # Save results
        watersheds.to_file(output_watershed_path)
        print(f"✓ Saved watershed results: {output_watershed_path}")
        
        print("SBEVA model execution completed successfully!")
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
            print(f"SBEVA analysis failed: {e}")
    except Exception as e:
        raise RuntimeError(f"SBEVA model execution failed: {str(e)}")
        
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
# ===================================================================================================================
# ====================== Main SBEVA Analysis Function - Memory Optimized
# ====================================================================================================================
import sys
from contextlib import redirect_stdout
def run_sbeva_analysis(stream_polyline_path, ws_shapefile_path,
                      original_atlas14_24hr_100yr_asci_files_dir_path, temp_dir,
                      dem_UTM_path, AOI_PIDF_24hr_100yr_output_raster_path,
                      AOI_PIDF_24hr_100yr_output_categorized_raster_path,
                      PRISM_ppt_30yr_normal_800mM4_raster_path,
                      PRISM_AOI_30yr_precip_Normal_resampled_UTM_path,
                      PRISM_AOI_categorized_30yr_precip_Normal_resampled_UTM_path,
                      PRISM_tmean_30yr_normal_800mM4_raster_path,
                      PRISM_AOI_30yr_tmean_Normal_resampled_UTM_path,
                      PRISM_AOI_categorized_30yr_tmean_Normal_resampled_UTM_path,
                      PRISM_solclear_30yr_normal_800mM4_raster_path,
                      PRISM_AOI_30yr_solclear_Normal_resampled_UTM_path,
                      PRISM_AOI_categorized_30yr_solclear_Normal_resampled_UTM_path,
                      nlcd_2021_conus_raster_path, base_raster_path,
                      usa_states_shapefile_path, GSSURGO_soil_data_output_dir,
                      output_slope_percentage_AOI_raster_path,
                      categorized_output_slope_percentage_AOI_raster_path,
                      nlcd_erosion_susceptibility_table_csv_path,
                      output_aoi_nlcd_raster_path, output_aoi_categorized_nlcd_raster_path,
                      sbeva_output_weighted_raster_path, sbeva_output_watershed_path,
                      output_stream_buffer_path, variables_list: List[str] = ['hydgrp', 'rootznaws','drainagecl', 'runoff'],
                      sbeva_weights_dict: Dict[str, float] = None, stream_buffer_distance: float = None,
                      user_id=None,
                      project_name=None,
                      task_type=None,
                      check_cancellation_func=None):
    """
    Complete SBEVA analysis workflow.
    
    This function orchestrates the entire SBEVA (Stream-Based Environmental Vulnerability Assessment) 
    analysis using template-based processing to minimize memory usage and eliminate redundant operations.
    
    Parameters:
    -----------
    
    stream_polyline_path : str
        Path to stream polyline shapefile
    ws_shapefile_path : str
        Path to watershed polygon shapefile
    original_atlas14_24hr_100yr_asci_files_dir_path : str
        Directory containing NOAA Atlas 14 ASCII files
    temp_dir : str
        Temporary directory for intermediate files
    dem_UTM_path : str
        Path to DEM raster in UTM projection
    [... other parameters remain the same as original ...]
    sbeva_weights_dict : Dict[str, float]
        Dictionary containing weights for all SBEVA variables
    stream_buffer_distance : float
        Buffer distance around streams in meters
        
    Returns:
    --------
    None
        Results are written to output files
    """
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
        print(f"SBEVA analysis failed: {e}")
    # Setup log file
    log_file_path = os.path.join(os.path.dirname(sbeva_output_watershed_path), 'processing_log.txt')
    
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
        print("STARTING SBEVA ANALYSIS WORKFLOW")
        print("=" * 80)
        
        if sbeva_weights_dict is None:
            raise ValueError("sbeva_weights_dict must be provided")
        
        if stream_buffer_distance is None:
            stream_buffer_distance = 20.0  # Default 20 meters
        
        # Create centralized temporary directory
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="sbeva_analysis_")
            cleanup_temp = True
        else:
            cleanup_temp = False
            os.makedirs(temp_dir, exist_ok=True)
        
        try:        
            # Create output directories
            output_paths = [
                AOI_PIDF_24hr_100yr_output_raster_path,
                categorized_output_slope_percentage_AOI_raster_path,
                output_aoi_nlcd_raster_path,
                sbeva_output_weighted_raster_path,
                sbeva_output_watershed_path
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
                print(f"SBEVA analysis failed: {e}")
            for output_path in output_paths:
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
                    method_prefix="sbeva"
                )
                print("✓ STEP 1 COMPLETED: Precipitation data processed")
            except Exception as e:
                raise RuntimeError(f"Step 1 failed - Precipitation processing: {str(e)}")
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
                print(f"SBEVA analysis failed: {e}")
            # Step 2: Process PRISM climate data
            print("\n" + "=" * 50)
            print("STEP 2: Processing PRISM climate data")
            print("=" * 50)
            
            try:
                # Process precipitation normal
                print("Processing PRISM precipitation normal...")
                process_prism_raster(
                    PRISM_ppt_30yr_normal_800mM4_raster_path,
                    dem_UTM_path,
                    ws_shapefile_path, 
                    PRISM_AOI_30yr_precip_Normal_resampled_UTM_path,
                    PRISM_AOI_categorized_30yr_precip_Normal_resampled_UTM_path,
                    temp_dir, var='precip'
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
                    print(f"SBEVA analysis failed: {e}")
                
                # Process temperature normal
                print("Processing PRISM temperature normal...")
                process_prism_raster(
                    PRISM_tmean_30yr_normal_800mM4_raster_path,
                    dem_UTM_path,
                    ws_shapefile_path, 
                    PRISM_AOI_30yr_tmean_Normal_resampled_UTM_path,
                    PRISM_AOI_categorized_30yr_tmean_Normal_resampled_UTM_path,
                    temp_dir, var='temp'
                )
                
                # Process solar radiation normal
                print("Processing PRISM solar radiation normal...")
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
                    print(f"SBEVA analysis failed: {e}")
                process_prism_raster(
                    PRISM_solclear_30yr_normal_800mM4_raster_path,
                    dem_UTM_path,
                    ws_shapefile_path, 
                    PRISM_AOI_30yr_solclear_Normal_resampled_UTM_path,
                    PRISM_AOI_categorized_30yr_solclear_Normal_resampled_UTM_path,
                    temp_dir, var='solar'
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
                    print(f"SBEVA analysis failed: {e}")
                
                print("✓ STEP 2 COMPLETED: PRISM climate data processed")
            except Exception as e:
                raise RuntimeError(f"Step 2 failed - PRISM processing: {str(e)}")
            
            # Step 3: Process GSSURGO soil data
            print("\n" + "=" * 50)
            print("STEP 3: Processing GSSURGO soil data")
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
                print(f"SBEVA analysis failed: {e}")
            try:
                soil_rasters = gssurgo_to_variable_rasters(
                    base_raster_path, 
                    ws_shapefile_path,
                    dem_UTM_path, 
                    GSSURGO_soil_data_output_dir,
                    variables_list, 
                    usa_states_shapefile_path,
                    temp_dir=temp_dir, 
                    state_abbr_column="stusps"
                )
                print(f"✓ STEP 3 COMPLETED: {len(soil_rasters)} soil variables processed")
            except Exception as e:
                raise RuntimeError(f"Step 3 failed - GSSURGO processing: {str(e)}")
            
            
            # Step 4: Calculate and categorize slope
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
                print(f"SBEVA analysis failed: {e}")
            print("\n" + "=" * 50)
            print("STEP 4: Processing slope data")
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
                    
                print("✓ STEP 4 COMPLETED: Slope data processed")
            except Exception as e:
                raise RuntimeError(f"Step 4 failed - Slope processing: {str(e)}")
            
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
                print(f"SBEVA analysis failed: {e}")
            # Step 5: Process NLCD with erosion
            print("\n" + "=" * 50)
            print("STEP 5: Processing NLCD and erosion data")
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
                print(f"SBEVA analysis failed: {e}")
            try:
                nlcd_results = process_nlcd_with_erosion(
                    ws_shapefile_path, 
                    dem_UTM_path, 
                    nlcd_2021_conus_raster_path,
                    nlcd_erosion_susceptibility_table_csv_path, 
                    output_aoi_nlcd_raster_path, 
                    output_aoi_categorized_nlcd_raster_path,
                    temp_dir=temp_dir
                )
                print("✓ STEP 5 COMPLETED: NLCD and erosion data processed")
            except Exception as e:
                raise RuntimeError(f"Step 5 failed - NLCD processing: {str(e)}")
            
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
                print(f"SBEVA analysis failed: {e}")
            # Step 6: Prepare SBEVA input paths
            print("\n" + "=" * 50)
            print("STEP 6: Preparing SBEVA input paths")
            print("=" * 50)
            
            # Map soil data outputs to SBEVA variable paths
            sbeva_input_paths = {
                'sbevaAWS_path': os.path.join(GSSURGO_soil_data_output_dir, 'rootznaws_raster.tif'),
                'sbevaRunoffClass_path': os.path.join(GSSURGO_soil_data_output_dir, 'runoff_raster.tif'),
                'sbevaDrainageClass_path': os.path.join(GSSURGO_soil_data_output_dir, 'drainagecl_raster.tif'),
                'sbevaHydroSoilGrp_path': os.path.join(GSSURGO_soil_data_output_dir, 'hydgrp_raster.tif')
            }
            
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
                print(f"SBEVA analysis failed: {e}")
            # Validate SBEVA input files
            missing_sbeva_inputs = []
            for var_name, file_path in sbeva_input_paths.items():
                if not os.path.exists(file_path):
                    missing_sbeva_inputs.append(f"{var_name}: {file_path}")
            
            if missing_sbeva_inputs:
                print("Warning: Some GSSURGO-derived inputs are missing:")
                for missing in missing_sbeva_inputs:
                    print(f"  - {missing}")
                print("SBEVA model will proceed with available inputs")
            
            print("✓ STEP 6 COMPLETED: SBEVA input paths configured")
            
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
                print(f"SBEVA analysis failed: {e}")
            # Step 7: Run SBEVA model
            print("\n" + "=" * 50)
            print("STEP 7: Running SBEVA model")
            print("=" * 50)
            
            try:
                run_sbeva_model(
                    dem_UTM_path,
                    AOI_PIDF_24hr_100yr_output_categorized_raster_path,
                    PRISM_AOI_categorized_30yr_precip_Normal_resampled_UTM_path,
                    PRISM_AOI_categorized_30yr_tmean_Normal_resampled_UTM_path,
                    PRISM_AOI_categorized_30yr_solclear_Normal_resampled_UTM_path,
                    sbeva_input_paths['sbevaAWS_path'],
                    sbeva_input_paths['sbevaRunoffClass_path'],
                    sbeva_input_paths['sbevaDrainageClass_path'],
                    sbeva_input_paths['sbevaHydroSoilGrp_path'],
                    categorized_output_slope_percentage_AOI_raster_path,
                    output_aoi_categorized_nlcd_raster_path,
                    stream_polyline_path,
                    ws_shapefile_path,
                    sbeva_output_weighted_raster_path,
                    sbeva_output_watershed_path,
                    output_stream_buffer_path,
                    weights_dict=sbeva_weights_dict,
                    stream_buffer_distance=stream_buffer_distance,
                    temp_dir=temp_dir,user_id = user_id,
                    project_name=project_name,task_type=task_type,
                    check_cancellation_func=check_cancellation_func
                )
                print("✓ STEP 7 COMPLETED: SBEVA model executed successfully")
            except Exception as e:
                raise RuntimeError(f"Step 7 failed - SBEVA model execution: {str(e)}")
            
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
                print(f"SBEVA analysis failed: {e}")
            # Final summary
            print("\n" + "=" * 80)
            print("SBEVA ANALYSIS WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            print("Final Outputs:")
            print(f"  ✓ Weighted Vulnerability Raster: {sbeva_output_weighted_raster_path}")
            print(f"  ✓ Watershed Vulnerability Results: {sbeva_output_watershed_path}")
            print(f"  ✓ Stream Buffer Shapefile: {output_stream_buffer_path}")
            
            print("\nIntermediate Products Created:")
            print(f"  ✓ Precipitation (PIDF): {AOI_PIDF_24hr_100yr_output_categorized_raster_path}")
            print(f"  ✓ PRISM Precipitation: {PRISM_AOI_categorized_30yr_precip_Normal_resampled_UTM_path}")
            print(f"  ✓ PRISM Temperature: {PRISM_AOI_categorized_30yr_tmean_Normal_resampled_UTM_path}")
            print(f"  ✓ PRISM Solar: {PRISM_AOI_categorized_30yr_solclear_Normal_resampled_UTM_path}")
            print(f"  ✓ Soil Data Directory: {GSSURGO_soil_data_output_dir}")
            print(f"  ✓ Slope: {categorized_output_slope_percentage_AOI_raster_path}")
            print(f"  ✓ NLCD: {output_aoi_categorized_nlcd_raster_path}")
            
      
            
            print("=" * 80)
            
        except Exception as e:
            print("\n" + "=" * 80)
            print("ERROR IN SBEVA ANALYSIS WORKFLOW")
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
            
            
