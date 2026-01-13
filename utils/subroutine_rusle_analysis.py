# utils/subroutine_rusle_analysis
# =================================================================================================
# 📦 System & File Operations
# =================================================================================================
import os
import glob
import shutil
import tempfile
import warnings
# =================================================================================================
# 📊 Data Analysis & Scientific Computing
# =================================================================================================
import numpy as np
# =================================================================================================
# 🧪 Geospatial Analysis
# =================================================================================================
import geopandas as gpd
from shapely.geometry import mapping, box
import whitebox

# =================================================================================================
# 🗺️ Raster & Vector Data Handling
# =================================================================================================
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.windows import from_bounds
from osgeo import gdal
from pyproj import CRS
# =================================================================================================
# 🖼️ Visualization & Plotting
# =================================================================================================
import matplotlib.pyplot as plt

# =================================================================================================
# ⚙️ Whitebox Tools Initialization
# =================================================================================================
wbt = whitebox.WhiteboxTools()
# =============================================================================================================================
# HANDLER FUCNTIONS
# ============================================================================================================================
# =================================================================================================
# Cancel task checker
# =================================================================================================

# ========================================================= Step 1: EDA and visualization functions
# Quick data exploration function
def explore_raster_data(raster_path):
    """
    Print basic information about the raster for debugging and verification.
    Works correctly with both numeric NODATA values and NaN.
    """
    with rasterio.open(raster_path) as src:
        print(f"Raster path: {raster_path}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Count of bands: {src.count}")
        print(f"Nodata value: {src.nodata}")
        
        # Read the data
        data = src.read(1)
        
        # Filter valid data properly based on whether NODATA is NaN or a numeric value
        if np.isnan(src.nodata):
            # For NaN NODATA, use numpy's isfinite
            valid_data = data[np.isfinite(data)]
        else:
            # For numeric NODATA, use direct comparison
            valid_data = data[data != src.nodata]
        
        if len(valid_data) > 0:
            print("\nStatistics on valid data:")
            print(f"Min: {np.nanmin(valid_data)}")
            print(f"Max: {np.nanmax(valid_data)}")
            print(f"Mean: {np.nanmean(valid_data)}")
            print(f"Standard deviation: {np.nanstd(valid_data)}")
            
            # Print a histogram of values
            print("\nValue distribution:")
            hist, bins = np.histogram(valid_data, bins=10)
            for i in range(len(hist)):
                print(f"{bins[i]:.2f} to {bins[i+1]:.2f}: {hist[i]} pixels")
            
            # Print percentage of missing data
            total_pixels = data.size
            valid_pixels = len(valid_data)
            missing_pixels = total_pixels - valid_pixels
            missing_percent = (missing_pixels / total_pixels) * 100
            
            print(f"\nData coverage:")
            print(f"Total pixels: {total_pixels}")
            print(f"Valid pixels: {valid_pixels} ({100 - missing_percent:.2f}%)")
            print(f"Missing/NODATA pixels: {missing_pixels} ({missing_percent:.2f}%)")
        else:
            print("No valid data found in raster.")
            
# =============================================================================================================================
# Clip raster to polygon
# ============================================================================================================================
import os
import tempfile
import warnings
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from osgeo import gdal
import subprocess
import shutil

def clip_raster_to_polygon(raster_path: str, polygon_path: str, output_path: str, input_crs: str, temp_dir: str) -> None:
    """
    Clips a raster to a polygon boundary with improved error handling.
    
    Parameters:
    -----------
    raster_path : str
        Path to the input raster file
    polygon_path : str
        Path to the boundary polygon shapefile
    output_path : str
        Path where the clipped raster will be saved
    input_crs : str
        CRS string for the input data
    temp_dir : str
        Temporary directory path for intermediate files
    """
    
    # Check if input raster exists and is valid
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Input raster does not exist: {raster_path}")
    
    # Check if polygon exists
    if not os.path.exists(polygon_path):
        raise FileNotFoundError(f"Polygon file does not exist: {polygon_path}")
    
    # Test if raster can be opened
    try:
        with rasterio.open(raster_path) as test_src:
            # Try to read a small sample to verify the file is valid
            test_data = test_src.read(1, window=rasterio.windows.Window(0, 0, min(10, test_src.width), min(10, test_src.height)))
            print(f"Input raster validation passed: {raster_path}")
    except Exception as e:
        raise ValueError(f"Input raster is corrupted or cannot be read: {raster_path}. Error: {str(e)}")
    
    try:
        # Read and process polygon
        poly = gpd.read_file(polygon_path).to_crs(input_crs)
        
        # Fix invalid geometries (self-intersections, etc.)
        poly['geometry'] = poly['geometry'].buffer(0)
        
        # Use the provided temp_dir but create a unique shapefile name to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        temp_poly_dir = os.path.join(temp_dir, f"clip_poly_{unique_id}")
        os.makedirs(temp_poly_dir, exist_ok=True)
        
        try:
            temp_poly = os.path.join(temp_poly_dir, 'temp_polygon.shp')
            poly.to_file(temp_poly)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build gdalwarp command with better error handling
            cmd = [
                'gdalwarp', 
                '-cutline', temp_poly, 
                '-crop_to_cutline',
                '-of', 'GTiff', 
                '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES',
                '-dstnodata', 'nan',
                raster_path, 
                output_path
            ]
            
            print(f"Running gdalwarp command: {' '.join(cmd)}")
            
            # Run the command with better error capture
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode != 0:
                error_msg = f"gdalwarp failed with return code {result.returncode}\n"
                error_msg += f"Command: {' '.join(cmd)}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}\n"
                error_msg += f"Input raster: {raster_path}\n"
                error_msg += f"Output path: {output_path}\n"
                print(error_msg)
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            # Verify output was created and has data
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")
            
            # Quick validation of output
            try:
                with rasterio.open(output_path) as check_src:
                    test_data = check_src.read(1, window=rasterio.windows.Window(0, 0, min(10, check_src.width), min(10, check_src.height)))
                    print(f"Output raster validation passed: {output_path}")
            except Exception as e:
                print(f"Warning: Output raster may be corrupted: {str(e)}")
                
        finally:
            # Clean up only the temporary polygon directory, not the main temp_dir
            try:
                if os.path.exists(temp_poly_dir):
                    shutil.rmtree(temp_poly_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp polygon directory {temp_poly_dir}: {str(e)}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"gdalwarp command timed out after 5 minutes for {raster_path}")
    except Exception as e:
        if isinstance(e, (subprocess.CalledProcessError, FileNotFoundError, ValueError, RuntimeError)):
            raise  # Re-raise these specific exceptions
        else:
            raise RuntimeError(f"Unexpected error in clip_raster_to_polygon: {str(e)}")
    
# =============================================================================================================================
# Clip raster to polygon with buffer
# ============================================================================================================================
def clip_raster_to_polygon_with_buffer(raster_path: str, polygon_path: str, output_path: str, input_crs: str, buffer_dist: float = 0.0, temp_dir: str = None) -> None:
    """
    Clips a raster to a polygon boundary with optional buffer and improved error handling.
    
    Parameters:
    -----------
    raster_path : str
        Path to the input raster file
    polygon_path : str
        Path to the polygon shapefile
    output_path : str
        Path to save the clipped raster
    input_crs : str
        Target CRS for reprojection (the raster's CRS)
    buffer_dist : float, optional
        Buffer distance in meters to expand the polygon before clipping. Default is 0.0 (no buffer)
    temp_dir : str, optional
        Temporary directory path for intermediate files
    """
    
    # Check if input raster exists and is valid
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Input raster does not exist: {raster_path}")
    
    # Check if polygon exists
    if not os.path.exists(polygon_path):
        raise FileNotFoundError(f"Polygon file does not exist: {polygon_path}")
    
    # Test if raster can be opened
    try:
        with rasterio.open(raster_path) as test_src:
            # Try to read a small sample to verify the file is valid
            test_data = test_src.read(1, window=rasterio.windows.Window(0, 0, min(10, test_src.width), min(10, test_src.height)))
            print(f"Input raster validation passed: {raster_path}")
    except Exception as e:
        raise ValueError(f"Input raster is corrupted or cannot be read: {raster_path}. Error: {str(e)}")
    
    try:
        # Read polygon
        poly = gpd.read_file(polygon_path)
        
        # Fix invalid geometries (self-intersections, etc.)
        poly['geometry'] = poly['geometry'].buffer(0)
        
        # Apply buffer if specified - buffer in meters then reproject
        if buffer_dist > 0:
            print(f"Applying {buffer_dist}m buffer to polygon...")
            
            # Parse the target CRS to understand its units
            target_crs = CRS.from_string(input_crs)
            
            if target_crs.is_geographic:
                # Target CRS is geographic - need to buffer in UTM then convert
                print(f"Target CRS is geographic, buffering in UTM first...")
                utm_crs = poly.estimate_utm_crs()
                print(f"Using UTM CRS for buffering: {utm_crs}")
                
                # Buffer in UTM (meters), then convert to target CRS
                poly_utm = poly.to_crs(utm_crs)
                poly_utm['geometry'] = poly_utm['geometry'].buffer(buffer_dist)
                poly = poly_utm.to_crs(input_crs)
                
            else:
                # Target CRS is projected - check its units
                try:
                    # Try to get unit information
                    crs_units = target_crs.axis_info[0].unit_name if target_crs.axis_info else 'metre'
                    print(f"Target CRS units: {crs_units}")
                    
                    if crs_units.lower() in ['metre', 'meter', 'm']:
                        # Units are meters - reproject then buffer directly
                        print(f"Target CRS uses meters, reprojecting then buffering {buffer_dist}m...")
                        poly = poly.to_crs(input_crs)
                        poly['geometry'] = poly['geometry'].buffer(buffer_dist)
                        
                    elif crs_units.lower() in ['foot', 'feet', 'ft', 'us survey foot']:
                        # Units are feet - convert buffer distance
                        buffer_feet = buffer_dist * 3.28084
                        print(f"Target CRS uses feet, reprojecting then buffering {buffer_feet:.2f} feet...")
                        poly = poly.to_crs(input_crs)
                        poly['geometry'] = poly['geometry'].buffer(buffer_feet)
                        
                    else:
                        # Unknown units - buffer in UTM then convert
                        print(f"Unknown target CRS units ({crs_units}), buffering in UTM first...")
                        utm_crs = poly.estimate_utm_crs()
                        poly_utm = poly.to_crs(utm_crs)
                        poly_utm['geometry'] = poly_utm['geometry'].buffer(buffer_dist)
                        poly = poly_utm.to_crs(input_crs)
                        
                except Exception as e:
                    # Fallback: buffer in UTM then convert
                    print(f"Could not determine CRS units ({e}), buffering in UTM first...")
                    utm_crs = poly.estimate_utm_crs()
                    poly_utm = poly.to_crs(utm_crs)
                    poly_utm['geometry'] = poly_utm['geometry'].buffer(buffer_dist)
                    poly = poly_utm.to_crs(input_crs)
        else:
            # No buffer - just reproject
            poly = poly.to_crs(input_crs)
        
        # Handle temp directory - use provided one or create with proper cleanup management
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="clip_buffer_temp_")
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        # Use unique naming to avoid conflicts between parallel processes
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        temp_poly_dir = os.path.join(temp_dir, f"clip_poly_buffer_{unique_id}")
        os.makedirs(temp_poly_dir, exist_ok=True)
        
        try:
            temp_poly = os.path.join(temp_poly_dir, 'temp_polygon.shp')
            poly.to_file(temp_poly)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build gdalwarp command with better error handling
            cmd = [
                'gdalwarp', 
                '-cutline', temp_poly, 
                '-crop_to_cutline',
                '-of', 'GTiff', 
                '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES',
                '-dstnodata', 'nan',
                raster_path, 
                output_path
            ]
            
            print(f"Running gdalwarp command: {' '.join(cmd)}")
            
            # Run the command with better error capture and timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode != 0:
                error_msg = f"gdalwarp failed with return code {result.returncode}\n"
                error_msg += f"Command: {' '.join(cmd)}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}\n"
                error_msg += f"Input raster: {raster_path}\n"
                error_msg += f"Output path: {output_path}\n"
                error_msg += f"Buffer distance: {buffer_dist}m\n"
                print(error_msg)
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            # Verify output was created and has data
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file was not created: {output_path}")
            
            # Quick validation of output
            try:
                with rasterio.open(output_path) as check_src:
                    test_data = check_src.read(1, window=rasterio.windows.Window(0, 0, min(10, check_src.width), min(10, check_src.height)))
                    print(f"Successfully clipped raster to: {output_path}")
            except Exception as e:
                print(f"Warning: Output raster may be corrupted: {str(e)}")
                
        finally:
            # Clean up only the temporary polygon directory, not the main temp_dir
            try:
                if os.path.exists(temp_poly_dir):
                    shutil.rmtree(temp_poly_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp polygon directory {temp_poly_dir}: {str(e)}")
            
            # Only clean up the main temp directory if we created it
            if cleanup_temp and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up main temp directory {temp_dir}: {str(e)}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"gdalwarp command timed out after 5 minutes for {raster_path}")
    except Exception as e:
        if isinstance(e, (subprocess.CalledProcessError, FileNotFoundError, ValueError, RuntimeError)):
            raise  # Re-raise these specific exceptions
        else:
            raise RuntimeError(f"Unexpected error in clip_raster_to_polygon_with_buffer: {str(e)}")


# ================================================================================================================================================
# Extract the states abbreviations that the region - ploygon intersects with
# ===========================================================================================================================================
import os
import time
import warnings
import tempfile
import subprocess
import shutil
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from osgeo import gdal

def get_us_states_crossed(polygon_path, usa_states_shapefile_path, state_abbr_column="stusps"):
    """
    Finds the US state abbreviations that the given boundary polygon crosses.

    Args:
        polygon_path (str): Path to the zipped shapefile containing the boundary polygon.
        usa_states_shapefile_path (str): Path to a known US states shapefile (should contain a state abbreviation column).
        state_abbr_column (str): Column name in the US states shapefile for state abbreviations (default 'stusps').

    Returns:
        list: A list of US state abbreviation strings that the boundary crosses.
    """
    try:
        # Load the user's boundary shapefile and convert to WGS84 (EPSG:4326)
        boundary_gdf = gpd.read_file(f"{polygon_path}").to_crs("EPSG:4326")

        # Load the US states shapefile and convert to WGS84 (EPSG:4326)
        states_gdf = gpd.read_file(f"zip://{usa_states_shapefile_path}").to_crs("EPSG:4326")

        # Create a unified boundary geometry (in case there are multiple features)
        boundary_union = boundary_gdf.geometry.unary_union

        # Find all states that intersect with the boundary polygon
        intersecting_states = states_gdf[states_gdf.intersects(boundary_union)]

        if intersecting_states.empty:
            return []

        if state_abbr_column not in intersecting_states.columns:
            print(f"Column '{state_abbr_column}' not found in the states shapefile. Available columns: {intersecting_states.columns.tolist()}")
            return []

        # Extract the unique state abbreviations from the intersecting states
        state_abbr_list = intersecting_states[state_abbr_column].unique().tolist()
        return state_abbr_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return []   

# =============================================================================================================================
# Step 2 Calculate R-factor: Rainfall Erosivity Factor
# (a) Process 100yr 30min PIDF from NOAA Atlas14 
# ============================================================================================================================
import uuid
import tempfile
import os

def process_noaa_atlas14_for_boundary(atlas14_dir, boundary_shapefile, dem_path, output_raster_path, 
                                      categorized_raster_path=None, temp_dir=None, method_prefix="rusle"):
    """
    Process NOAA Atlas 14 precipitation data for a given boundary and resample to match DEM resolution.
    
    Parameters:
    -----------
    atlas14_dir : str
        Path to directory containing NOAA Atlas 14 data organized in subdirectories
    boundary_shapefile : str
        Path to the boundary shapefile
    dem_path : str
        Path to the DEM file used for resampling resolution
    output_raster_path : str
        Path where the final resampled raster will be saved
    categorized_raster_path : str, optional
        Path where the categorized raster (values 1-5) will be saved
    temp_dir : str, optional
        Path to temporary directory. If None, creates a unique one
    method_prefix : str, optional
        Prefix for temporary files to distinguish between methods (default: "rusle")
    
    Returns:
    --------
    str
        Path to the resulting raster file or a message if no data found
    """
    # Generate unique identifier for this process
    process_id = str(uuid.uuid4())[:8]
    
    # Use provided temp_dir or create one with unique naming
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix=f"{method_prefix}_{process_id}_")
        cleanup_temp = True
    else:
        # Ensure the provided temp_dir is unique for this process
        temp_dir = os.path.join(temp_dir, f"{method_prefix}_{process_id}")
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False
    
    # Read the boundary shapefile
    try:
        boundary_gdf = gpd.read_file(boundary_shapefile)
        boundary_crs = boundary_gdf.crs
        
        # If there are multiple geometries, dissolve them
        if len(boundary_gdf) > 1:
            boundary_gdf = boundary_gdf.dissolve()
            
    except Exception as e:
        return f"Error reading boundary shapefile: {str(e)}"
    
    # Read the DEM to get its properties
    try:
        with rasterio.open(dem_path) as dem_src:
            dem_crs = dem_src.crs
            dem_transform = dem_src.transform
            dem_width = dem_src.width
            dem_height = dem_src.height
            dem_bounds = dem_src.bounds
            
            print(f"DEM properties - CRS: {dem_crs}, Width: {dem_width}, Height: {dem_height}")
            print(f"DEM resolution: {dem_transform[0]} x {abs(dem_transform[4])}")
    except Exception as e:
        return f"Error reading DEM: {str(e)}"
    
    # Find all ASC files in the directory structure
    asc_files = []
    for dirpath, _, _ in os.walk(atlas14_dir):
        asc_files.extend(glob.glob(os.path.join(dirpath, "*.asc")))
    
    if not asc_files:
        return "Error: No ASC files found in the specified directory"
        
    print(f"Found {len(asc_files)} ASC files")
    
    # Create a temporary directory for processing
    try:
        # Find ASC files that intersect with the DEM area
        intersecting_files = []
        
        for asc_file in asc_files:
            try:
                # Check for corresponding .prj file to get CRS
                prj_file = asc_file.replace('.asc', '.prj')
                if not os.path.exists(prj_file):
                    print(f"Skipping {asc_file} - no .prj file found")
                    continue
                    
                # Open the ASC file
                with rasterio.open(asc_file) as src:
                    # Reproject DEM bounds to ASC CRS to check intersection
                    dem_bounds_in_asc_crs = rasterio.warp.transform_bounds(
                        dem_crs, src.crs, *dem_bounds
                    )
                    
                    # Create a bounding box for the DEM in ASC CRS
                    dem_box = box(*dem_bounds_in_asc_crs)
                    
                    # Create a bounding box for the ASC file
                    asc_box = box(*src.bounds)
                    
                    # Check for intersection
                    if dem_box.intersects(asc_box):
                        intersecting_files.append(asc_file)
                        print(f"ASC file {os.path.basename(asc_file)} intersects with DEM")
            except Exception as e:
                print(f"Error checking intersection for {asc_file}: {str(e)}")
                continue
        
        if not intersecting_files:
            return "No ASC files found that intersect with the DEM area"
            
        print(f"Found {len(intersecting_files)} ASC files that intersect with the DEM area")
        
        # Process each intersecting ASC file - clip to DEM bounds
        cropped_rasters = []
        
        for i, asc_file in enumerate(intersecting_files):
            try:
                # Use unique naming with process_id and file index
                temp_cropped = os.path.join(temp_dir, f"{method_prefix}_cropped_{process_id}_{i}_{os.path.basename(asc_file)}.tif")
                
                with rasterio.open(asc_file) as src:
                    # Create a buffered DEM bounds (500m buffer)
                    buffered_dem_bounds = (
                        dem_bounds[0] - 500,  # minx - 500m
                        dem_bounds[1] - 500,  # miny - 500m
                        dem_bounds[2] + 500,  # maxx + 500m
                        dem_bounds[3] + 500   # maxy + 500m
                    )
                    
                    # Convert buffered DEM bounds to ASC CRS
                    dem_bounds_in_asc_crs = rasterio.warp.transform_bounds(
                        dem_crs, src.crs, *buffered_dem_bounds
                    )
                    
                    # Create a window from the buffered bounds
                    window = from_bounds(*dem_bounds_in_asc_crs, src.transform)
                    
                    # Read the data in the window
                    out_image = src.read(1, window=window)
                    
                    # Get the transform for the window
                    out_transform = rasterio.windows.transform(window, src.transform)
                    
                    # Handle nodata values and convert to float32
                    if np.issubdtype(src.dtypes[0], np.integer):
                        # For integer types
                        temp_nodata = src.nodata if src.nodata is not None else -9999
                        out_image = out_image.astype(np.float32)
                        out_image = np.where(out_image == temp_nodata, np.nan, out_image)
                    else:
                        # For float types, ensure it's float32
                        out_image = out_image.astype(np.float32)
                        if src.nodata is not None and not np.isnan(src.nodata):
                            out_image = np.where(out_image == src.nodata, np.nan, out_image)
                    
                    # Update metadata
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[0],
                        "width": out_image.shape[1],
                        "transform": out_transform,
                        "nodata": np.nan,
                        "dtype": "float32",
                        "count": 1
                    })
                    
                    # Write the cropped raster
                    with rasterio.open(temp_cropped, "w", **out_meta) as dest:
                        dest.write(out_image, 1)
                    
                    # Verify we have valid data
                    with rasterio.open(temp_cropped) as check:
                        check_data = check.read(1)
                        valid_data = check_data[~np.isnan(check_data)]
                        if len(valid_data) > 0:
                            cropped_rasters.append(temp_cropped)
                            print(f"Successfully cropped {os.path.basename(asc_file)}")
                        else:
                            print(f"Skipping {os.path.basename(asc_file)} - no valid data after cropping")
            except Exception as e:
                print(f"Error cropping {asc_file}: {str(e)}")
                continue
        
        if not cropped_rasters:
            return "Failed to crop any ASC files to the DEM bounds"
            
        # Merge cropped rasters if multiple exist
        if len(cropped_rasters) > 1:
            # First reproject all to DEM CRS
            reprojected_rasters = []
            
            for j, cropped_raster in enumerate(cropped_rasters):
                temp_reprojected = os.path.join(temp_dir, f"{method_prefix}_reprojected_{process_id}_{j}_{os.path.basename(cropped_raster)}")
                
                with rasterio.open(cropped_raster) as src:
                    if src.crs != dem_crs:
                        # Get transform for reprojection
                        dst_transform, dst_width, dst_height = calculate_default_transform(
                            src.crs, dem_crs, src.width, src.height, *src.bounds)
                        
                        # Update metadata
                        dst_kwargs = src.meta.copy()
                        dst_kwargs.update({
                            'crs': dem_crs,
                            'transform': dst_transform,
                            'width': dst_width,
                            'height': dst_height,
                            'nodata': np.nan,
                            'dtype': "float32"
                        })
                        
                        # Create reprojected raster
                        with rasterio.open(temp_reprojected, 'w', **dst_kwargs) as dst:
                            # Initialize destination array
                            dst_array = np.full((dst_kwargs['count'], dst_height, dst_width), 
                                               np.nan, dtype=np.float32)
                            
                            # Reproject
                            for i in range(1, src.count + 1):
                                reproject(
                                    source=rasterio.band(src, i),
                                    destination=dst_array[i-1],
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=dst_transform,
                                    dst_crs=dem_crs,
                                    resampling=Resampling.nearest
                                )
                            
                            dst.write(dst_array)
                        
                        reprojected_rasters.append(temp_reprojected)
                    else:
                        # Already in DEM CRS
                        reprojected_rasters.append(cropped_raster)
            
            # Merge reprojected rasters with unique naming
            merged_raster = os.path.join(temp_dir, f"{method_prefix}_noaa_merged_{process_id}.tif")
            
            # Open all reprojected rasters
            rasters_to_merge = []
            for raster_path in reprojected_rasters:
                try:
                    raster = rasterio.open(raster_path)
                    data = raster.read(1)
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        rasters_to_merge.append(raster)
                except Exception as e:
                    print(f"Error opening {raster_path} for merge: {str(e)}")
                    continue
            
            if not rasters_to_merge:
                return "Failed to prepare any rasters for merging"
            
            try:
                # Perform the merge
                mosaic, out_transform = merge(rasters_to_merge)
                
                # Get metadata from first raster
                out_meta = rasters_to_merge[0].meta.copy()
                
                # Update metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "crs": dem_crs,
                    "nodata": np.nan
                })
                
                # Write merged raster
                with rasterio.open(merged_raster, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                # Close all rasters
                for raster in rasters_to_merge:
                    raster.close()
                    
                print("Successfully merged rasters")
                
            except Exception as e:
                return f"Error merging rasters: {str(e)}"
        else:
            # Only one cropped raster
            with rasterio.open(cropped_rasters[0]) as src:
                if src.crs != dem_crs:
                    # Need to reproject to DEM CRS
                    merged_raster = os.path.join(temp_dir, f"{method_prefix}_noaa_merged_{process_id}.tif")
                    
                    # Get transform for reprojection
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, dem_crs, src.width, src.height, *src.bounds)
                    
                    # Update metadata
                    dst_kwargs = src.meta.copy()
                    dst_kwargs.update({
                        'crs': dem_crs,
                        'transform': dst_transform,
                        'width': dst_width,
                        'height': dst_height,
                        'nodata': np.nan,
                        'dtype': "float32"
                    })
                    
                    # Create reprojected raster
                    with rasterio.open(merged_raster, 'w', **dst_kwargs) as dst:
                        # Initialize destination array
                        dst_array = np.full((dst_kwargs['count'], dst_height, dst_width), 
                                           np.nan, dtype=np.float32)
                        
                        # Reproject
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=dst_array[i-1],
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=dst_transform,
                                dst_crs=dem_crs,
                                resampling=Resampling.nearest
                            )
                        
                        dst.write(dst_array)
                else:
                    # Already in DEM CRS, just copy
                    merged_raster = os.path.join(temp_dir, f"{method_prefix}_noaa_merged_{process_id}.tif")
                    shutil.copy(cropped_rasters[0], merged_raster)
        
        # Now resample to exactly match DEM resolution using bilinear interpolation
        try:
            # Create resampled raster with unique naming
            resampled_raster = os.path.join(temp_dir, f"{method_prefix}_resampled_{process_id}.tif")
            
            with rasterio.open(merged_raster) as src, rasterio.open(dem_path) as dem_src:
                # Use DEM as a template for the output
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                    'crs': dem_crs,
                    'transform': dem_transform,
                    'width': dem_width,
                    'height': dem_height,
                    'nodata': np.nan,
                    'dtype': "float32"
                })
                
                # Resample using bilinear interpolation
                with rasterio.open(resampled_raster, 'w', **dst_kwargs) as dst:
                    # Initialize destination array
                    dst_array = np.full((dst_kwargs['count'], dem_height, dem_width), 
                                       np.nan, dtype=np.float32)
                    
                    # For each band
                    for i in range(1, src.count + 1):
                        # Reproject and resample
                        reproject(
                            source=rasterio.band(src, i),
                            destination=dst_array[i-1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dem_transform,
                            dst_crs=dem_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan
                        )
                    
                    # Write to output
                    dst.write(dst_array)
            
            print("Successfully resampled to match DEM resolution")
                
        except Exception as e:
            return f"Error during resampling: {str(e)}"
        
        # Finally, clip the resampled raster to the boundary shapefile (without buffer)
        try:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
            
            with rasterio.open(resampled_raster) as src:
                # Convert boundary to DEM CRS if needed
                if boundary_crs != dem_crs:
                    boundary_to_use = boundary_gdf.to_crs(dem_crs)
                else:
                    boundary_to_use = boundary_gdf
                
                # Get geometry for masking
                geometries = [mapping(geom) for geom in boundary_to_use.geometry]
                
                # Mask the raster with the boundary
                out_image, out_transform = mask(src, geometries, crop=True, all_touched=True, nodata=np.nan)
                
                # Update metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": np.nan
                })
                
                # Write the final clipped and resampled raster
                with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)
            
            # Verify the final output
            with rasterio.open(output_raster_path) as check:
                check_data = check.read(1)
                valid_data = check_data[~np.isnan(check_data)]
                if len(valid_data) == 0:
                    return "Error: Final output raster has no valid data"
                
                print(f"Final output stats: min={valid_data.min()}, max={valid_data.max()}, mean={valid_data.mean()}")
                print(f"Output dimensions: {check.width}x{check.height}")
                print(f"Output resolution: {check.transform[0]}x{abs(check.transform[4])}")
                
                # Add softmax-based normalization to categorize values between 1 and 5
                if categorized_raster_path:
                    try:
                        # Read the raster data
                        with rasterio.open(output_raster_path) as src:
                            data = src.read(1)
                            mask_data = ~np.isnan(data)
                            
                            # Extract the valid data for processing
                            valid_data = data[mask_data]
                            
                            if len(valid_data) > 0:
                                # Apply softmax-based normalization only to valid data
                                # First normalize to avoid numerical issues
                                normalized = (valid_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data) + 1e-10)
                                
                                # Calculate softmax (adjust the temperature parameter as needed)
                                temperature = 0.1  # Adjust this value to control the spread
                                exp_values = np.exp(normalized / temperature)
                                softmax_values = exp_values / (np.sum(exp_values) + 1e-10)
                                
                                # Scale to 1-5 range
                                # Create 5 equal bins from the values and assign integer categories
                                bins = np.linspace(np.min(valid_data), np.max(valid_data), 6)
                                categorized_values = np.digitize(valid_data, bins)
                                
                                # Ensure values are between 1 and 5 (digitize can produce values from 1 to 6)
                                categorized_values = np.clip(categorized_values, 1, 5)
                                
                                # Create output array with original NaN values preserved
                                output_categorized = np.full_like(data, np.nan, dtype=np.float32)
                                output_categorized[mask_data] = categorized_values
                            else:
                                print("Warning: No valid data found for categorization")
                                output_categorized = np.full_like(data, np.nan, dtype=np.float32)
                            
                            # Update metadata for categorized raster
                            cat_meta = src.meta.copy()
                            cat_meta.update({
                                "dtype": "float32",
                                "nodata": np.nan,
                                "driver": "GTiff"
                            })
                            
                            # Make sure the output directory exists
                            os.makedirs(os.path.dirname(categorized_raster_path), exist_ok=True)
                            
                            # Write the categorized raster
                            with rasterio.open(categorized_raster_path, 'w', **cat_meta) as dst:
                                dst.write(output_categorized, 1)
                                
                            print(f"Successfully created categorized raster: {categorized_raster_path}")
                            print(f"Category distribution: {np.bincount(categorized_values, minlength=6)[1:]}")
                            
                    except Exception as e:
                        print(f"Error creating categorized raster: {str(e)}")
                        # Continue with the process even if categorization fails
            
            return output_raster_path
                
        except Exception as e:
            return f"Error clipping to boundary: {str(e)}"
        
    finally:
        # Only clean up if we created the temp directory
        if cleanup_temp:
            shutil.rmtree(temp_dir)


# Usage examples:
# For RUSLE method:
# result = process_noaa_atlas14_for_boundary(atlas14_dir, boundary_shapefile, dem_path, 
#                                           output_raster_path, categorized_raster_path, 
#                                           method_prefix="rusle")




# ==================================================================================================================
# (b) Calculate R -factor
# ===================================================================================================================
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def calculate_r_factor(pi30_raster_path, output_path):
    """
    Note on units and conversions:
    
    Input: 
    - PI30 values are in inches*1000
    - For example, a value of 4500 represents 4.5 inches of precipitation in 30 minutes
    
    Conversion:
    - Inches to mm: 1 inch = 25.4 mm
    - 30-min to hourly rate: multiply by 2
    - So 4.5 inches in 30 minutes = 4.5 * 25.4 * 2 = 228.6 mm/hr
    
    Output:
    - R-factor (Ri) = KEi × PI30i
    - Where KEi is in MJ ha^-1 hr^-1 mm^-1 and PI30i is in mm/hr
    - So Ri is in MJ ha^-1 hr^-1
    
    ------------------------------------------------------------------------------------------------------------
    Calculate R-factor raster based on NOAA-Atlas14 30-min 100-yr precipitation intensity (PI30) raster.
    
    The calculation follows the modified RUSLE model (Panda et al., 2022):
    Ri = KEi × PI30i
    
    Where KEi is calculated as:
    KEi = 0.119 + 0.0873 × log10(PI30i) for PI30i ≤ 76 mm/hr
    KEi = 0.283 MJ ha^-1 hr^-1 mm^-1 for PI30i > 76 mm/hr
    
    Input precipitation data is assumed to be in inches*1000 and will be converted to mm/hr
    for the calculations.
    
    Parameters:
    -----------
    pi30_raster_path : str
        Path to the NOAA-Atlas14 30-min 100-yr precipitation intensity (PI30) raster (.tif) file
        Values are assumed to be in inches*1000
    output_path : str
        Path to save the resulting R-factor raster
        
    Returns:
    --------
    str
        Path to the processed output R-factor raster
    """
    print(f"Calculating R-factor using PI30 raster: {pi30_raster_path}")
    
    # Open the PI30 raster file
    with rasterio.open(pi30_raster_path) as src:
        # Read the data
        pi30_data = src.read(1)
        
        # Get the metadata for output
        meta = src.meta.copy()
        
        # Create mask for valid data (non-nodata values)
        if src.nodata is not None:
            valid_mask = (pi30_data != src.nodata)
        else:
            # If no nodata value is specified, assume all data is valid
            valid_mask = np.ones_like(pi30_data, dtype=bool)
            
        # Convert precipitation from inches*1000 to mm/hr
        # 1 inch = 25.4 mm, and for 30-min values to hourly, multiply by 2
        # So the conversion is: (value/1000) * 25.4 * 2 = value * 0.0508
        conversion_factor = 0.0508  # (1/1000) * 25.4 * 2
        
        # Apply conversion only to valid data
        pi30_mm_hr = np.zeros_like(pi30_data, dtype=np.float32)
        pi30_mm_hr[valid_mask] = pi30_data[valid_mask] * conversion_factor
        
        # Print original and converted values for debugging
        if np.any(valid_mask):
            print(f"Original PI30 data (inches*1000): min={np.min(pi30_data[valid_mask])}, max={np.max(pi30_data[valid_mask])}")
            print(f"Converted PI30 data (mm/hr): min={np.min(pi30_mm_hr[valid_mask])}, max={np.max(pi30_mm_hr[valid_mask])}")
        
        # Replace original data with converted data for further calculations
        pi30_data = pi30_mm_hr
        
        # Initialize array for KE values with same shape as pi30_data
        ke_data = np.zeros_like(pi30_data, dtype=np.float32)
        
        # Calculate KE values based on the provided equation
        # For PI30i ≤ 76 mm/hr: KEi = 0.119 + 0.0873 × log10(PI30i)
        # For PI30i > 76 mm/hr: KEi = 0.283 MJ ha^-1 hr^-1 mm^-1
        
        # Handle potential zeros or negative values in PI30 data
        low_mask = (pi30_data <= 0) & valid_mask
        if np.any(low_mask):
            print(f"Warning: {np.sum(low_mask)} pixels have PI30 values <= 0 mm/hr. Setting their KE values to 0.")
            ke_data[low_mask] = 0
        
        # Calculate KE for 0 < PI30i ≤ 76 mm/hr
        mid_mask = (pi30_data > 0) & (pi30_data <= 76) & valid_mask
        if np.any(mid_mask):
            ke_data[mid_mask] = 0.119 + 0.0873 * np.log10(pi30_data[mid_mask])
        
        # Set KE = 0.283 for PI30i > 76 mm/hr
        high_mask = (pi30_data > 76) & valid_mask
        if np.any(high_mask):
            ke_data[high_mask] = 0.283
        
        # Set nodata values in KE array
        if src.nodata is not None:
            ke_data[~valid_mask] = src.nodata
        
        # Calculate R-factor: Ri = KEi × PI30i
        r_factor_data = ke_data * pi30_data
        
        # Preserve nodata values
        if src.nodata is not None:
            r_factor_data[~valid_mask] = src.nodata
        
        # Print summary statistics for debugging
        if np.any(valid_mask):
            print(f"PI30 data range: min={np.min(pi30_data[valid_mask])}, max={np.max(pi30_data[valid_mask])}")
            print(f"KE data range: min={np.min(ke_data[valid_mask])}, max={np.max(ke_data[valid_mask])}")
            print(f"R-factor data range: min={np.min(r_factor_data[valid_mask])}, max={np.max(r_factor_data[valid_mask])}")
        
        # Update metadata for output
        meta.update({
            'dtype': 'float32',
            'driver': 'GTiff',
            'nodata': src.nodata if src.nodata is not None else -9999
        })
        
        # Write the output R-factor raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(r_factor_data.astype(np.float32), 1)
        
        print(f"R-factor calculation complete. Output saved to: {output_path}")
        return output_path


# # Example usage:
# if __name__ == "__main__":
#     # Example paths - replace with your actual file paths
#     pi30_raster_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/NOAA_Atlas14_100yr_30min_UTM.tif"
#     output_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/Rfactor_UTM.tif"
    
#     calculate_r_factor(pi30_raster_path, output_path)
    

# =====================================================================================================================================    
# Step 3 K-factor: Extract and resample kf-factor raster from GSSURGO
# ===============================================================================================================================
# ================================================================================================================================================
# Extract the states abbreviations that the region - ploygon intersects with
# ===========================================================================================================================================
import os
import time
import warnings
import tempfile
import subprocess
import shutil
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from osgeo import gdal

def wait_for_file(file_path, max_wait=120, check_interval=2):
    """
    Wait for file to appear and be fully written to disk.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to wait for
    max_wait : int
        Maximum time to wait in seconds (default: 120)
    check_interval : int
        Time between checks in seconds (default: 2)
        
    Returns:
    --------
    bool
        True if file is ready, False if timeout
    """
    wait_time = 0
    last_size = 0
    stable_count = 0
    
    print(f"Waiting for file to be written: {os.path.basename(file_path)}")
    
    while wait_time < max_wait:
        if os.path.exists(file_path):
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 3:  # File size stable for 3 checks
                        print(f"✅ File ready: {current_size:,} bytes")
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
                    if current_size > 0:
                        print(f"📝 Writing... {current_size:,} bytes")
            except OSError:
                # File might be locked, continue waiting
                pass
        else:
            print(f"⏳ Waiting for file to appear... ({wait_time}s)")
        
        time.sleep(check_interval)
        wait_time += check_interval
    
    print(f"❌ Timeout after {max_wait}s")
    return False

def force_file_sync(file_path):
    """Force file system synchronization and verify file exists"""
    try:
        # Force filesystem sync
        os.sync()
        time.sleep(1)
        
        # Try to verify file is readable
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read first byte
            return True
        return False
    except Exception as e:
        print(f"Warning: File sync verification failed: {e}")
        return os.path.exists(file_path)

# =====================================================================================================================================    
# Step 3 K-factor: Extract and resample kf-factor raster from GSSURGO
# ===============================================================================================================================
def gssurgo_to_kffactor_raster(gssurgo_soil_data_directory_path,
                                boundary_shp_path,
                                dem_path,
                                usa_states_shapefile_path,
                                output_raster_path,
                                temp_dir=None):
    """
    Creates a K-factor raster by reading pre-existing gSSURGO rasters, clipping
    them to a boundary, and aligning them with a reference DEM.

    This function dynamically determines which state(s) the boundary crosses,
    mosaics the corresponding 'kffact.tif' files if necessary, and processes
    the result.

    Parameters:
    -----------
    gssurgo_soil_data_directory_path : str
        Path to the base directory containing gSSURGO state folders
        (e.g., '.../instance/core_data/Soil_GSSURGO').
    boundary_shp_path : str
        Path to the boundary shapefile used for clipping.
    dem_path : str
        Path to the reference DEM for resolution and final CRS.
    usa_states_shapefile_path : str
        Path to the zipped US States shapefile for state identification.
    output_raster_path : str
        Path where the final output K-factor raster will be saved.

    Returns:
    --------
    str
        Path to the created K-factor raster, or None if failed.
    """
    print("Starting gSSURGO K-factor raster creation...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    # Use provided temp_dir instead of creating one
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    else:
        cleanup_temp = False
    try:
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: Find which states the boundary crosses
        print("\n=== Step 1: State Identification ===")
        print("Determining which states the boundary crosses...")
        state_abbrs = get_us_states_crossed(boundary_shp_path, usa_states_shapefile_path)
        if not state_abbrs:
            print("❌ Error: Could not determine state for the given boundary. Aborting.")
            return None
        print(f"✅ Boundary crosses the following states: {state_abbrs}")

        # Step 2: Locate the 'kffact.tif' raster(s) for the identified states
        print("\n=== Step 2: Locate K-factor Rasters ===")
        kffact_raster_paths = []
        for state in state_abbrs:
            # Path format is like: .../Soil_GSSURGO/SC/kffact.tif
            potential_path = os.path.join(gssurgo_soil_data_directory_path, state.upper(), 'kffact.tif')
            if os.path.exists(potential_path):
                kffact_raster_paths.append(potential_path)
                print(f"✅ Found K-factor raster for {state}: {potential_path}")
            else:
                print(f"⚠️  Warning: kffact.tif not found for state '{state}' at {potential_path}")

        if not kffact_raster_paths:
            print("❌ Error: No 'kffact.tif' files found for the identified states. Aborting.")
            return None

        # Step 3: Mosaic rasters if the boundary crosses multiple states
        print("\n=== Step 3: Mosaic Rasters (if needed) ===")
        source_kffact_raster = ""
        if len(kffact_raster_paths) == 1:
            print("✅ Single state detected. Using raster directly.")
            source_kffact_raster = kffact_raster_paths[0]
        else:
            print("🔗 Multiple states detected. Mosaicking 'kffact.tif' files...")
            source_kffact_raster = os.path.join(temp_dir, "rusle_kffact_mosaic.tif")
            
            try:
                gdal.Warp(
                    source_kffact_raster,
                    kffact_raster_paths,
                    format='GTiff',
                    options=['COMPRESS=LZW', 'TILED=YES']
                )
                
                # Wait for mosaic file to be written
                if wait_for_file(source_kffact_raster, max_wait=60):
                    print(f"✅ Mosaicked raster created at: {source_kffact_raster}")
                else:
                    print("❌ Error: Mosaic creation timeout")
                    return None
                    
            except Exception as e:
                print(f"❌ Error during mosaicking: {e}")
                return None

        # Step 4: Clip the source k-factor raster to the boundary shapefile
        print("\n=== Step 4: Clip to Boundary ===")
        print("Clipping k-factor raster to the boundary...")
        clipped_raster_path = os.path.join(temp_dir, "rusle_kffact_clipped.tif")
        
        try:
            input_crs = 'EPSG:5070'
            
            # Import the clipping function (assuming it's available)
            clip_raster_to_polygon(source_kffact_raster, boundary_shp_path, clipped_raster_path, input_crs, temp_dir)
            
            
            # Fine-tune clipping with rasterio for exact polygon boundary
            with rasterio.open(clipped_raster_path) as src:
                poly_gdf = gpd.read_file(boundary_shp_path).to_crs(src.crs)
                out_image, out_transform = rasterio.mask.mask(
                    src, poly_gdf.geometry, crop=True, filled=True, 
                    nodata=src.nodata if src.nodata else np.nan
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1], 
                    "width": out_image.shape[2], 
                    "transform": out_transform
                })
                
                # Write refined clipped raster
                refined_clipped_path = os.path.join(temp_dir, "rusle_kffact_clipped_refined.tif")
                with rasterio.open(refined_clipped_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                # Force sync and verify
                force_file_sync(refined_clipped_path)
                clipped_raster_path = refined_clipped_path
                
            print(f"✅ Clipped raster saved to: {clipped_raster_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during clipping: {e}. This might happen if the boundary is outside the raster extent.")
            return None
        except Exception as e:
            print(f"❌ Unexpected error during clipping: {e}")
            return None

        # Step 5: Resample and reproject the clipped raster to match the DEM
        print("\n=== Step 5: Resample to Match DEM ===")
        print("Resampling clipped raster to match the reference DEM...")
        
        try:
            with rasterio.open(dem_path) as dem:
                dem_crs = dem.crs
                dem_res = dem.res
                print(f"📏 Target resolution: {dem_res[0]:.2f} x {dem_res[1]:.2f}")
                print(f"🗺️  Target CRS: {dem_crs}")

            # Use local temp file first, then copy to final location
            temp_output = os.path.join(temp_dir, "rusle_kffact_final_temp.tif")
            
            gdal.Warp(
                temp_output,
                clipped_raster_path,
                format='GTiff',
                dstSRS=dem_crs.to_wkt(),
                xRes=dem_res[0],
                yRes=dem_res[1],
                resampleAlg=gdal.GRA_Bilinear,  # Bilinear is good for continuous data
                dstNodata=np.nan,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            
            # Wait for temp file to be ready
            if not wait_for_file(temp_output, max_wait=120):
                print("❌ Error: Resampling timeout")
                return None
            
            # Copy to final location
            print(f"📋 Copying to final location: {output_raster_path}")
            shutil.copy2(temp_output, output_raster_path)
            
            # Force sync and verify final file
            force_file_sync(output_raster_path)
            
            if not wait_for_file(output_raster_path, max_wait=60):
                print("❌ Error: Final file copy timeout")
                return None
                
        except Exception as e:
            print(f"❌ Error during resampling: {e}")
            return None

        # Step 6: Final verification
        print("\n=== Step 6: Final Verification ===")
        try:
            if os.path.exists(output_raster_path):
                file_size = os.path.getsize(output_raster_path)
                print(f"✅ File created successfully: {output_raster_path}")
                print(f"📁 File size: {file_size:,} bytes")
                
                # Test if it's readable with rasterio
                with rasterio.open(output_raster_path) as src:
                    print(f"📊 Raster properties:")
                    print(f"   - Dimensions: {src.width} x {src.height}")
                    print(f"   - Bands: {src.count}")
                    print(f"   - CRS: {src.crs}")
                    print(f"   - Resolution: {src.res}")
                    print(f"   - Data type: {src.dtypes[0]}")
                    
                    # Check for valid data
                    sample_data = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
                    valid_pixels = np.sum(np.isfinite(sample_data))
                    print(f"   - Valid pixels in sample: {valid_pixels}/{sample_data.size}")
                
                print("🎉 K-factor raster creation completed successfully!")
                return output_raster_path
                
            else:
                print(f"❌ Final file verification failed: {output_raster_path} does not exist")
                return None
                
        except Exception as e:
            print(f"❌ Error during final verification: {e}")
            return None
    finally:
        # Only clean up if we created the temp directory
        if cleanup_temp:
            shutil.rmtree(temp_dir)
# # Example usage:
# if __name__ == "__main__":
#     result = gssurgo_to_kffactor_raster(
#         gssurgo_soil_data_directory_path='/home/smdsgit/Culvert_socket/Culvert_web_app/instance/core_data/Soil_GSSURGO',
#         boundary_shp_path='/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/final_flag_removed_ws_polygon_filtered_by_area_UTM.shp',
#         dem_path='/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/DEM_UTM.tif',
#         usa_states_shapefile_path='/home/smdsgit/Culvert_socket/Culvert_web_app/instance/core_data/US States and Territories Shapefile_20250216.zip',
#         output_raster_path='/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/Kffactor_UTM.tif'
#     )
    
#     if result:
#         print(f"\n🎯 SUCCESS: K-factor raster available at {result}")
#     else:
#         print(f"\n💥 FAILED: K-factor raster creation failed")
# =========================================================================================================================
# Step 4: Calculate LS factor in SI Unit - Memory Efficient Version with Input Clipping
# ========================================================================================================================

import numpy as np
import rasterio
import geopandas as gpd
import os
import tempfile
import subprocess
import shutil
from typing import Optional, Tuple, Iterator
import gc

def calculate_ls_factor(dem_path: str, flow_acc_path: str, output_path: str, 
                               shapefile_path: Optional[str] = None, 
                               chunk_size: int = 4096, temp_dir: str = None):
    """
    Calculate the LS factor for the Modified Revised Universal Soil Loss Equation (RUSLE)
    using a DEM and flow accumulation raster with memory-efficient chunking.
    
    This version clips input rasters first using the boundary shapefile, then processes
    the clipped rasters with chunking for memory efficiency.
    
    Parameters:
    -----------
    dem_path : str
        Path to the DEM raster file (GeoTIFF)
    flow_acc_path : str
        Path to the flow accumulation raster file (GeoTIFF)
    output_path : str
        Path where the output LS factor raster will be saved (GeoTIFF)
    shapefile_path : str, optional
        Path to a shapefile for clipping the input rasters (if provided)
    chunk_size : int, default=4096
        Size of chunks to process at once (pixels). Adjust based on available memory.
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Use provided temp_dir
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            cleanup_temp = True
        else:
            cleanup_temp = False
        # Initialize paths with original paths
        clipped_dem_path = dem_path
        clipped_flow_path = flow_acc_path
        # Get CRS from boundary shapefile and clip input rasters first
        if shapefile_path and os.path.exists(shapefile_path):
            print("Clipping input rasters to boundary shapefile...")
            
            # Get CRS from boundary shapefile
            boundary_gdf = gpd.read_file(shapefile_path)
            boundary_crs = boundary_gdf.crs.to_string()
            print(f"Using boundary CRS: {boundary_crs}")
            
            clipped_dem_path = os.path.join(temp_dir, "rusle_temp_dem1_clipped.tif")
            clipped_flow_path = os.path.join(temp_dir, "rusle_temp_flow_clipped.tif")
            
            try:
                # Clip DEM ====================================================
                print("Clipping DEM...")
                clip_raster_to_polygon(dem_path, shapefile_path, clipped_dem_path, boundary_crs, temp_dir)
                # Clip to exact polygon boundary using rasterio
                with rasterio.open(clipped_dem_path) as src:
                    poly_gdf = gpd.read_file(shapefile_path).to_crs(src.crs)
                    out_image, out_transform = rasterio.mask.mask(src, poly_gdf.geometry, crop=True, filled=True, nodata=src.nodata if src.nodata else np.nan)
                    out_meta = src.meta.copy()
                    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
                    with rasterio.open(clipped_dem_path, "w", **out_meta) as dest:
                        dest.write(out_image)
                print(f"Clipped DEM raster saved to: {clipped_dem_path}")
                # Clip flow accumulation =======================================
                print("Clipping flow accumulation...")
                clip_raster_to_polygon(flow_acc_path, shapefile_path, clipped_flow_path, boundary_crs, temp_dir)
                # Clip to exact polygon boundary using rasterio
                with rasterio.open(clipped_flow_path) as src:
                    poly_gdf = gpd.read_file(shapefile_path).to_crs(src.crs)
                    out_image, out_transform = rasterio.mask.mask(src, poly_gdf.geometry, crop=True, filled=True, nodata=src.nodata if src.nodata else np.nan)
                    out_meta = src.meta.copy()
                    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
                    with rasterio.open(clipped_flow_path, "w", **out_meta) as dest:
                        dest.write(out_image)
                print(f"Clipped Flow Accum raster saved to: {clipped_flow_path}")
                print("All Input rasters clipped successfully")
                
            except Exception as e:
                print(f"Error clipping input rasters: {str(e)}")
                # Clean up temp files if they exist
                for temp_file in [clipped_dem_path, clipped_flow_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                return False
        
        # Now process the (potentially clipped) rasters with chunking
        print("Starting LS factor calculation...")
        
        # Open input rasters to get metadata
        with rasterio.open(clipped_dem_path) as dem_src:
            profile = dem_src.profile.copy()
            rows, cols = dem_src.height, dem_src.width
            cell_size = dem_src.res[0]
            print(f"Processing raster dimensions: {rows}x{cols}, cell size: {cell_size}m")
            
        # Verify flow accumulation dimensions match
        with rasterio.open(clipped_flow_path) as flow_src:
            if flow_src.height != rows or flow_src.width != cols:
                print("Error: Flow accumulation and DEM rasters have different dimensions")
                return False
        
        # Update profile for output
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=np.nan,
            tiled=True,
            blockxsize=4096,
            blockysize=4096
        )
        
        # Create output raster and process in chunks
        with rasterio.open(output_path, 'w', **profile) as out_src:
            print(f"Processing {rows}x{cols} raster in {chunk_size}x{chunk_size} chunks...")
            
            processed_chunks = 0
            total_chunks = ((rows - 1) // chunk_size + 1) * ((cols - 1) // chunk_size + 1)
            
            # Process chunks
            for chunk_info in _get_chunks(rows, cols, chunk_size):
                row_start, col_start, chunk_rows, chunk_cols = chunk_info
                
                # Read DEM and flow accumulation chunks with padding for slope calculation
                dem_chunk, flow_chunk = _read_chunks_with_padding(
                    clipped_dem_path, clipped_flow_path, row_start, col_start, 
                    chunk_rows, chunk_cols, rows, cols
                )
                
                if dem_chunk is None or flow_chunk is None:
                    continue
                
                # Calculate LS factor for this chunk
                ls_chunk = _calculate_ls_chunk_rasterio(dem_chunk, flow_chunk, cell_size)
                
                # Create window for writing
                window = rasterio.windows.Window(
                    col_off=col_start, row_off=row_start,
                    width=chunk_cols, height=chunk_rows
                )
                
                # Write chunk to output
                out_src.write(ls_chunk, 1, window=window)
                
                processed_chunks += 1
                if processed_chunks % 10 == 0:
                    print(f"Processed {processed_chunks}/{total_chunks} chunks")
                
                # Clean up memory
                del dem_chunk, flow_chunk, ls_chunk
                gc.collect()
        
        print(f"LS factor raster successfully created at: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error calculating LS factor: {str(e)}")
        return False
    finally:
        # Only clean up if we created the temp directory
        if cleanup_temp:
            shutil.rmtree(temp_dir)

def _get_chunks(rows: int, cols: int, chunk_size: int) -> Iterator[Tuple[int, int, int, int]]:
    """Generate chunk coordinates (row_start, col_start, chunk_rows, chunk_cols)"""
    for row in range(0, rows, chunk_size):
        for col in range(0, cols, chunk_size):
            chunk_rows = min(chunk_size, rows - row)
            chunk_cols = min(chunk_size, cols - col)
            yield (row, col, chunk_rows, chunk_cols)


def _read_chunks_with_padding(dem_path: str, flow_acc_path: str, 
                             row_start: int, col_start: int, 
                             chunk_rows: int, chunk_cols: int,
                             total_rows: int, total_cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read DEM and flow accumulation chunks with padding for slope calculation"""
    try:
        # Calculate padded bounds for DEM (needed for slope calculation)
        pad_size = 1
        dem_row_start = max(0, row_start - pad_size)
        dem_col_start = max(0, col_start - pad_size)
        dem_row_end = min(total_rows, row_start + chunk_rows + pad_size)
        dem_col_end = min(total_cols, col_start + chunk_cols + pad_size)
        
        dem_pad_rows = dem_row_end - dem_row_start
        dem_pad_cols = dem_col_end - dem_col_start
        
        # Read padded DEM chunk
        dem_window = rasterio.windows.Window(
            col_off=dem_col_start, row_off=dem_row_start,
            width=dem_pad_cols, height=dem_pad_rows
        )
        
        with rasterio.open(dem_path) as dem_src:
            dem_chunk = dem_src.read(1, window=dem_window)
        
        # Read flow accumulation chunk (no padding needed)
        flow_window = rasterio.windows.Window(
            col_off=col_start, row_off=row_start,
            width=chunk_cols, height=chunk_rows
        )
        
        with rasterio.open(flow_acc_path) as flow_src:
            flow_chunk = flow_src.read(1, window=flow_window)
        
        return dem_chunk, flow_chunk
        
    except Exception as e:
        print(f"Error reading chunks: {str(e)}")
        return None, None


def _calculate_ls_chunk_rasterio(dem_chunk: np.ndarray, flow_chunk: np.ndarray, 
                                cell_size: float) -> np.ndarray:
    """Calculate LS factor for a chunk"""
    try:
        # Calculate slope from padded DEM chunk
        if dem_chunk.shape[0] < 3 or dem_chunk.shape[1] < 3:
            # Handle edge case for small chunks
            slope_chunk = np.zeros(flow_chunk.shape, dtype=np.float32)
        else:
            # Calculate gradients using central difference
            dz_dx = (dem_chunk[1:-1, 2:] - dem_chunk[1:-1, :-2]) / (2 * cell_size)
            dz_dy = (dem_chunk[2:, 1:-1] - dem_chunk[:-2, 1:-1]) / (2 * cell_size)
            
            # Calculate slope
            slope_chunk = np.sqrt(dz_dx**2 + dz_dy**2)
            
            # Ensure slope chunk matches flow chunk dimensions
            target_shape = flow_chunk.shape
            if slope_chunk.shape != target_shape:
                # Resize slope chunk to match flow chunk
                from_rows, from_cols = slope_chunk.shape
                to_rows, to_cols = target_shape
                
                # Simple cropping/padding to match dimensions
                if from_rows >= to_rows and from_cols >= to_cols:
                    # Crop if slope chunk is larger
                    row_start = (from_rows - to_rows) // 2
                    col_start = (from_cols - to_cols) // 2
                    slope_chunk = slope_chunk[row_start:row_start+to_rows, 
                                            col_start:col_start+to_cols]
                else:
                    # Pad if slope chunk is smaller (shouldn't happen normally)
                    padded_slope = np.zeros(target_shape, dtype=np.float32)
                    padded_slope[:from_rows, :from_cols] = slope_chunk
                    slope_chunk = padded_slope
        
        # Convert slope to angle in radians
        slope_radians = np.arctan(slope_chunk)
        
        # Calculate slope percentage for m factor determination
        slope_percent = np.tan(slope_radians) * 100
        
        # Initialize m values based on slope percentage
        m = np.zeros_like(slope_percent, dtype=np.float32)
        m[slope_percent < 1] = 0.2
        m[(slope_percent >= 1) & (slope_percent < 3)] = 0.3
        m[(slope_percent >= 3) & (slope_percent < 5)] = 0.4
        m[slope_percent >= 5] = 0.5
        
        # Calculate L factor
        flow_length = flow_chunk.astype(np.float32) * cell_size
        ref_length = 22.13
        
        # Calculate L factor with safe handling of zero/negative values
        valid_cells = (flow_length > 0)
        l_factor = np.ones_like(flow_length, dtype=np.float32)
        
        if np.any(valid_cells):
            l_factor[valid_cells] = np.power(
                flow_length[valid_cells] / ref_length, 
                m[valid_cells]
            )
        
        # Calculate S factor: 65.41sin²θ + 4.56sinθ + 0.065
        sin_slope = np.sin(slope_radians)
        s_factor = 65.41 * np.square(sin_slope) + 4.56 * sin_slope + 0.065
        
        # Calculate final LS factor
        ls_factor = l_factor * s_factor
        
        # Handle invalid values
        ls_factor = np.where(np.isfinite(ls_factor), ls_factor, np.nan)
        
        return ls_factor.astype(np.float32)
        
    except Exception as e:
        print(f"Error in LS calculation for chunk: {str(e)}")
        # Return array of NaN values matching flow_chunk shape
        return np.full(flow_chunk.shape, np.nan, dtype=np.float32)


# # Example usage with memory monitoring
# if __name__ == "__main__":
#     import psutil
#     import time
    
#     # Monitor memory usage
#     process = psutil.Process()
#     initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
#     dem_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/DEM_UTM.tif"
#     flow_acc_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/D8flow_dir_UTM.tif"
#     output_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/LS_factor_UTM.tif"
#     shapefile_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/final_flag_removed_ws_polygon_filtered_by_area_UTM.shp"
    
#     start_time = time.time()
    
#     # Process with input clipping and chunking
#     success = calculate_ls_factor(
#         dem_path, flow_acc_path, output_path, shapefile_path, chunk_size=4096
#     )
    
#     end_time = time.time()
#     final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
#     print(f"\nProcessing completed successfully: {success}")
#     print(f"Processing time: {end_time - start_time:.2f} seconds")
#     print(f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB")
#     print(f"Peak memory increase: {final_memory - initial_memory:.1f} MB")


# ===========================================================================================================================
# Step 5: Calculate C-factor raster
# ==============================================================================================================================
## (A) NDVI Data processing 

# Data sources 

# https://earthexplorer.usgs.gov/ 

# https://doi.org/10.5066/P9QOEFNP
# ==========================================================================================================
# Required imports
import os
import tempfile
import warnings
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box, mapping
from pyproj import CRS
from osgeo import gdal
import subprocess
import shutil


def process_ndvi_raster(ndvi_path, dem_path, boundary_path, output_path=None, ndvi_scale_factor=10000.0, 
                        buffer_dist=0.0, chunk_size=1024, temp_dir=None):
    """
    Process an NDVI raster to match a DEM raster's projection and resolution, 
    with clipping to specified boundaries.
    
    Parameters:
    -----------
    ndvi_path : str
        Path to the NDVI raster (.tif) file
    dem_path : str
        Path to the DEM raster (.tif) file
    boundary_path : str
        Path to the boundary polygon shapefile
    output_path : str, optional
        Path to save the processed raster. If None, will create a path based on the input.
    ndvi_scale_factor : float, optional
        Scale factor to convert integer NDVI values to float range (-1 to 1).
        Default is 10000.0 for USGS eVIIRS NDVI products.
    buffer_dist : float, optional
        Buffer distance in meters to expand the boundary polygon before clipping.
        Default is 0.0 (no buffer).
    chunk_size : int, optional
        Size of chunks for memory-efficient processing. Default is 1024 pixels.
        Reduce if you encounter memory issues, increase for faster processing.
        
    Returns:
    --------
    str
        Path to the processed output raster
    """
    if output_path is None:
        basename = os.path.splitext(os.path.basename(ndvi_path))[0]
        output_path = f"{basename}_processed.tif"
    
    print(f"Processing NDVI raster: {ndvi_path}")
    print(f"Using DEM raster: {dem_path}")
    print(f"Using boundary shapefile: {boundary_path}")
    if buffer_dist > 0:
        print(f"Using buffer distance: {buffer_dist} meters")
    
    # PRE-CLIPPING STEP: Clip DEM and NDVI to boundary first
    # Get CRS information
    boundary_gdf = gpd.read_file(boundary_path)
    boundary_crs = boundary_gdf.crs.to_string()
    
    # Get DEM CRS
    with rasterio.open(dem_path) as dem_src:
        dem_crs_string = dem_src.crs.to_string()
    
    # Define NDVI CRS (Lambert Azimuthal Equal Area - uses meters)
    ndvi_crs = CRS.from_proj4(
        "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    ndvi_crs_string = ndvi_crs.to_string()
    
    # Use provided temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    else:
        cleanup_temp = False
    try: 
        clipped_dem_path = os.path.join(temp_dir, "rusle_temp_dem2_clipped.tif")
        clipped_ndvi_path = os.path.join(temp_dir, "rusle_temp_ndvi_clipped.tif")
        
        # Clip DEM with buffer
        print("\n=== PRE-CLIPPING STEP ===")
        print("Clipping DEM...")
        clip_raster_to_polygon_with_buffer(dem_path, boundary_path, clipped_dem_path, dem_crs_string, buffer_dist,temp_dir)
        
        # Clip NDVI with buffer  
        print("Clipping NDVI...")
        clip_raster_to_polygon_with_buffer(ndvi_path, boundary_path, clipped_ndvi_path, ndvi_crs_string, buffer_dist,temp_dir)
        
        print("Pre-clipping complete!\n")
        
        # NOW USE YOUR ORIGINAL LOGIC WITH CLIPPED RASTERS
        print("=== ORIGINAL PROCESSING LOGIC ===")
        
        # Step 1: Read the clipped DEM raster to get target CRS and resolution
        with rasterio.open(clipped_dem_path) as dem_src:
            dem_crs = dem_src.crs
            dem_transform = dem_src.transform
            dem_bounds = dem_src.bounds
            dem_res = dem_src.res
            
            # Create a buffered bounding box from the DEM (500m buffer)
            buffer_distance = 500  # in the units of the DEM CRS
            # Convert buffer distance from meters to coordinate system units if needed
            # This is approximate and assumes UTM or similar projection where units are meters
            buffered_bounds = (
                dem_bounds.left - buffer_distance,
                dem_bounds.bottom - buffer_distance,
                dem_bounds.right + buffer_distance,
                dem_bounds.top + buffer_distance
            )
            buffered_box = box(*buffered_bounds)
        
        # Step 2: Reproject the clipped NDVI raster to match the DEM's CRS
        with rasterio.open(clipped_ndvi_path) as ndvi_src:
            # Calculate the transform for reprojection
            transform, width, height = calculate_default_transform(
                ndvi_src.crs, dem_crs, ndvi_src.width, ndvi_src.height, 
                *ndvi_src.bounds
            )
            
            # Create a temporary file for the reprojected raster
            reprojected_path = os.path.join(temp_dir, "rusle_ndvi_reprojected.tif")
            clipped_path = os.path.join(temp_dir, "rusle_ndvi_clipped_bounds.tif")
            resampled_path = os.path.join(temp_dir, "rusle_ndvi_resampled.tif")
            # Set up the output raster
            kwargs = ndvi_src.meta.copy()
            # Convert from integer to float format for NDVI with scaling factor
            is_converting_datatype = False
            if ndvi_scale_factor and not np.issubdtype(ndvi_src.dtypes[0], np.floating):
                is_converting_datatype = True
                kwargs.update({'dtype': 'float32'})
            
            kwargs.update({
                'crs': dem_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Update nodata value in the metadata, using np.nan for float datatypes
            if np.issubdtype(ndvi_src.dtypes[0], np.floating):
                kwargs.update({'nodata': np.nan})
            elif 'nodata' in kwargs and kwargs['nodata'] is not None:
                # Keep existing nodata value for non-float types
                pass
            else:
                # Set default nodata value for non-float types if none exists
                kwargs.update({'nodata': 0})
                
            # Perform reprojection with chunking for memory efficiency
            with rasterio.open(reprojected_path, 'w', **kwargs) as dst:
                # For each band (typically just one for NDVI)
                for i in range(1, ndvi_src.count + 1):
                    if ndvi_scale_factor and not np.issubdtype(ndvi_src.dtypes[0], np.floating):
                        print(f"Processing band {i} with chunking (chunk_size={chunk_size})...")
                        
                        # Get band dimensions
                        band_height, band_width = ndvi_src.height, ndvi_src.width
                        
                        # Calculate number of chunks
                        n_chunks_y = (band_height + chunk_size - 1) // chunk_size
                        n_chunks_x = (band_width + chunk_size - 1) // chunk_size
                        total_chunks = n_chunks_y * n_chunks_x
                        
                        print(f"  Processing {total_chunks} chunks ({n_chunks_y}x{n_chunks_x})...")
                        
                        # Create full output array for this band
                        output_data = np.full((band_height, band_width), np.nan, dtype=np.float32)
                        
                        chunk_count = 0
                        for y_chunk in range(n_chunks_y):
                            for x_chunk in range(n_chunks_x):
                                chunk_count += 1
                                
                                # Calculate chunk boundaries
                                y_start = y_chunk * chunk_size
                                y_end = min(y_start + chunk_size, band_height)
                                x_start = x_chunk * chunk_size
                                x_end = min(x_start + chunk_size, band_width)
                                
                                # Read chunk
                                window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                                chunk_data = ndvi_src.read(i, window=window).astype(np.float32)
                                
                                if chunk_count % 50 == 0 or chunk_count == total_chunks:
                                    print(f"    Processing chunk {chunk_count}/{total_chunks}")
                                
                                # Set nodata value
                                src_nodata = ndvi_src.nodata if ndvi_src.nodata is not None else 0
                                
                                # Create mask for valid data (non-nodata)
                                valid_mask = (chunk_data != src_nodata)
                                
                                # Apply scaling only to valid data
                                chunk_data[valid_mask] = chunk_data[valid_mask] / ndvi_scale_factor
                                
                                # Set nodata values
                                chunk_data[~valid_mask] = np.nan
                                
                                # Store processed chunk
                                output_data[y_start:y_end, x_start:x_end] = chunk_data
                        
                        # Show scaling results
                        valid_data = output_data[~np.isnan(output_data)]
                        if len(valid_data) > 0:
                            print(f"  Scaled data range: min={np.min(valid_data):.4f}, max={np.max(valid_data):.4f}")
                        
                        # Set final nodata value for reprojection
                        output_data[np.isnan(output_data)] = -9999.0
                        
                        # Create a temporary in-memory source for reproject
                        with rasterio.io.MemoryFile() as memfile:
                            with memfile.open(
                                driver='GTiff',
                                height=output_data.shape[0],
                                width=output_data.shape[1],
                                count=1,
                                dtype=output_data.dtype,
                                transform=ndvi_src.transform,
                                crs=ndvi_src.crs,
                                nodata=-9999.0
                            ) as temp_src:
                                temp_src.write(output_data, 1)
                                
                                reproject(
                                    source=rasterio.band(temp_src, 1),
                                    destination=rasterio.band(dst, i),
                                    src_transform=ndvi_src.transform,
                                    src_crs=ndvi_src.crs,
                                    dst_transform=transform,
                                    dst_crs=dem_crs,
                                    resampling=Resampling.bilinear,
                                    src_nodata=-9999.0,
                                    dst_nodata=-9999.0
                                )
                    else:
                        # No scaling needed - process normally
                        reproject(
                            source=rasterio.band(ndvi_src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=ndvi_src.transform,
                            src_crs=ndvi_src.crs,
                            dst_transform=transform,
                            dst_crs=dem_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=ndvi_src.nodata,
                            dst_nodata=kwargs.get('nodata')
                        )
        
        # Step 3: Clip the reprojected NDVI to the buffered DEM extent
        with rasterio.open(reprojected_path) as src:
            # Create a mask using the buffered bounding box
            geoms = [mapping(buffered_box)]
            out_image, out_transform = mask(src, geoms, crop=True, nodata=-9999.0 if np.issubdtype(src.dtypes[0], np.floating) else src.nodata)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": -9999.0 if np.issubdtype(src.dtypes[0], np.floating) else src.nodata
            })
            
            # Write the clipped raster
            with rasterio.open(clipped_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        # Step 4: Resample the clipped NDVI to match the DEM resolution (with chunking)
        with rasterio.open(clipped_path) as src:
            # Calculate scaling factors
            scale_factor_x = dem_res[0] / src.res[0]
            scale_factor_y = dem_res[1] / src.res[1]
            
            # Calculate new dimensions
            new_width = int(src.width / scale_factor_x)
            new_height = int(src.height / scale_factor_y)
            
            print(f"Resampling from {src.width}x{src.height} to {new_width}x{new_height}")
            
            # Calculate the new transform
            new_transform = rasterio.transform.from_bounds(
                src.bounds.left, src.bounds.bottom, 
                src.bounds.right, src.bounds.top, 
                new_width, new_height
            )
            
            # For large rasters, use chunked reading for resampling
            if src.width * src.height > chunk_size * chunk_size * 4:
                print(f"Large raster detected, using chunked resampling...")
                
                # Calculate output chunks
                out_chunk_size = max(4096, chunk_size // 2)  # Smaller chunks for output
                n_chunks_y = (new_height + out_chunk_size - 1) // out_chunk_size
                n_chunks_x = (new_width + out_chunk_size - 1) // out_chunk_size
                total_chunks = n_chunks_y * n_chunks_x
                
                print(f"  Processing {total_chunks} output chunks...")
                
                # Update metadata for the resampled raster
                resampled_meta = src.meta.copy()
                resampled_meta.update({
                    'transform': new_transform,
                    'width': new_width,
                    'height': new_height,
                    'nodata': -9999.0 if np.issubdtype(src.dtypes[0], np.floating) else src.nodata
                })
                
                
                # Write chunked resampled data
                with rasterio.open(resampled_path, 'w', **resampled_meta) as dst:
                    chunk_count = 0
                    for y_chunk in range(n_chunks_y):
                        for x_chunk in range(n_chunks_x):
                            chunk_count += 1
                            
                            if chunk_count % 20 == 0 or chunk_count == total_chunks:
                                print(f"    Resampling chunk {chunk_count}/{total_chunks}")
                            
                            # Calculate output chunk boundaries
                            y_start = y_chunk * out_chunk_size
                            y_end = min(y_start + out_chunk_size, new_height)
                            x_start = x_chunk * out_chunk_size
                            x_end = min(x_start + out_chunk_size, new_width)
                            
                            # Calculate corresponding input window
                            # Scale back to input coordinates
                            in_x_start = int(x_start * scale_factor_x)
                            in_x_end = min(int(x_end * scale_factor_x) + 1, src.width)
                            in_y_start = int(y_start * scale_factor_y)
                            in_y_end = min(int(y_end * scale_factor_y) + 1, src.height)
                            
                            # Read input chunk with some padding
                            input_window = rasterio.windows.Window(
                                in_x_start, in_y_start, 
                                in_x_end - in_x_start, in_y_end - in_y_start
                            )
                            
                            # Read and resample chunk
                            chunk_data = src.read(
                                out_shape=(src.count, y_end - y_start, x_end - x_start),
                                window=input_window,
                                resampling=Resampling.bilinear
                            )
                            
                            # Write chunk to output
                            output_window = rasterio.windows.Window(
                                x_start, y_start, x_end - x_start, y_end - y_start
                            )
                            dst.write(chunk_data, window=output_window)
            else:
                # Small raster - process normally
                print("Small raster, processing normally...")
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                
                # Update metadata for the resampled raster
                resampled_meta = src.meta.copy()
                resampled_meta.update({
                    'transform': new_transform,
                    'width': new_width,
                    'height': new_height,
                    'nodata': -9999.0 if np.issubdtype(src.dtypes[0], np.floating) else src.nodata
                })
                
                # Create a temporary file for the resampled raster
                resampled_path = f"{os.path.splitext(output_path)[0]}_resampled.tif"
                
                # Write the resampled raster
                with rasterio.open(resampled_path, 'w', **resampled_meta) as dst:
                    dst.write(data)
            
            # Convert nodata values for floating point data types (applies to both methods)
            if np.issubdtype(src.dtypes[0], np.floating):
                print("Converting nodata values for floating point data...")
                with rasterio.open(resampled_path, 'r+') as dst:
                    for i in range(1, dst.count + 1):
                        # Read the entire band and process in memory chunks
                        band_data = dst.read(i)
                        band_data = np.where(band_data == dst.nodata, -9999.0, band_data)
                        dst.write(band_data, i)
        
        # Step 5: Final clip using the boundary shapefile (with chunking for large files)
        with rasterio.open(resampled_path) as src:
            # Read the boundary shapefile
            boundary_gdf = gpd.read_file(boundary_path)
            
            # Ensure the boundary has the same CRS as the raster
            if boundary_gdf.crs != src.crs:
                boundary_gdf = boundary_gdf.to_crs(src.crs)
            
            # Extract the geometries and create a mask
            geoms = [mapping(geom) for geom in boundary_gdf.geometry]
            
            # Check if we need chunking for final clipping
            if src.width * src.height > chunk_size * chunk_size * 4:
                print("Large raster detected, using chunked final clipping...")
                
                # Use rasterio's mask with chunking
                out_image, out_transform = mask(
                    src, geoms, crop=True, 
                    nodata=-9999.0 if np.issubdtype(src.dtypes[0], np.floating) else src.nodata,
                    all_touched=False, invert=False
                )
            else:
                print("Small raster, processing final clip normally...")
                out_image, out_transform = mask(
                    src, geoms, crop=True, 
                    nodata=-9999.0 if np.issubdtype(src.dtypes[0], np.floating) else src.nodata
                )
            
            # Update metadata for final output
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": np.nan if np.issubdtype(src.dtypes[0], np.floating) else src.nodata
            })
            
            # Write the final output (with chunking for large arrays)
            with rasterio.open(output_path, "w", **out_meta) as dest:
                if np.issubdtype(out_meta["dtype"], np.floating):
                    # Process in chunks to convert -9999.0 to np.nan
                    if out_image.size > chunk_size * chunk_size * 4:
                        print("Large output, using chunked writing...")
                        for i in range(out_image.shape[0]):
                            # Process band in chunks
                            band_data = out_image[i]
                            chunk_h = chunk_size
                            chunk_w = chunk_size
                            
                            for y in range(0, band_data.shape[0], chunk_h):
                                for x in range(0, band_data.shape[1], chunk_w):
                                    y_end = min(y + chunk_h, band_data.shape[0])
                                    x_end = min(x + chunk_w, band_data.shape[1])
                                    
                                    chunk = band_data[y:y_end, x:x_end].copy()
                                    chunk = np.where(chunk == -9999.0, np.nan, chunk)
                                    
                                    window = rasterio.windows.Window(x, y, x_end - x, y_end - y)
                                    dest.write(chunk, i + 1, window=window)
                    else:
                        # Small output - process normally
                        for i in range(out_image.shape[0]):
                            out_image[i] = np.where(out_image[i] == -9999.0, np.nan, out_image[i])
                        dest.write(out_image)
                else:
                    dest.write(out_image)
        
        print(f"Processing complete. Output saved to: {output_path}")
        return output_path
    
    finally:
        # Only clean up if we created the temp directory
        if cleanup_temp:
            shutil.rmtree(temp_dir)

# process_ndvi_raster(
#     ndvi_path="/home/smdsgit/Culvert_socket/Culvert_web_app/instance/core_data/US_eVSH_NDVI.2025.091-097.3KM.COMPRES.001.2025100164341/US_eVSH_NDVI.2025.091-097.3KM.VI_NDVI.001.2025100163611.tif",
#     dem_path="/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/DEM_UTM.tif",
#     boundary_path="/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/WS_deln/final_flag_removed_ws_polygon_filtered_by_area_UTM.shp",
#     output_path="/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/NDVI_AOI_UTM.tif",
#     ndvi_scale_factor=10000.0,
#     buffer_dist=500.0,  # 500 meter buffer
#     chunk_size=4096      # 4096x4096 pixel chunks for memory efficiency
# )
# =============================================================================================================================
## (B) Calculate C-factor raster
# ==============================================================================================================================
import os
import numpy as np
import rasterio

def calculate_cfactor_from_ndvi(ndvi_path, cfactor_path, alpha=2, beta=1, chunk_size=4096):
    """
    Calculate C-factor from NDVI using the Van der Knijff exponential decay relationship.
    
    The formula used is:
    C = e^((-alpha * NDVI)/(beta - NDVI))
    
    Where default values are alpha = 2 and beta = 1 (Van der Knijff et al., 2000)
    
    Parameters:
    -----------
    ndvi_path : str
        Path to the NDVI raster (.tif) file
    cfactor_path : str
        Path to save the resulting C-factor raster
    alpha : float, optional
        Alpha parameter in the Van der Knijff formula (default: 2)
    beta : float, optional
        Beta parameter in the Van der Knijff formula (default: 1)
    chunk_size : int, optional
        Size of chunks for memory-efficient processing. Default is 1024 pixels.
        Reduce if you encounter memory issues, increase for faster processing.
        
    Returns:
    --------
    str
        Path to the output C-factor raster
    """
    print(f"Calculating C-factor from NDVI: {ndvi_path}")
    print(f"Using chunk size: {chunk_size}x{chunk_size} pixels")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(cfactor_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the NDVI raster
    with rasterio.open(ndvi_path) as src:
        # Copy metadata from input to use for output
        out_meta = src.meta.copy()
        
        # Get nodata value from the NDVI raster
        nodata = src.nodata
        
        # Update metadata for output
        out_meta.update({
            'dtype': 'float32',
            'nodata': np.nan
        })
        
        # Get raster dimensions
        height, width = src.height, src.width
        
        # Calculate number of chunks
        n_chunks_y = (height + chunk_size - 1) // chunk_size
        n_chunks_x = (width + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_y * n_chunks_x
        
        # Check if chunking is needed
        if total_chunks > 1:
            print(f"Large raster detected, processing {total_chunks} chunks ({n_chunks_y}x{n_chunks_x})...")
        else:
            print("Small raster, processing as single chunk...")
        
        # Initialize statistics collection
        valid_cfactor_values = []
        total_valid_pixels = 0
        
        # Write the output C-factor raster with chunking
        with rasterio.open(cfactor_path, 'w', **out_meta) as dst:
            chunk_count = 0
            
            for y_chunk in range(n_chunks_y):
                for x_chunk in range(n_chunks_x):
                    chunk_count += 1
                    
                    # Calculate chunk boundaries
                    y_start = y_chunk * chunk_size
                    y_end = min(y_start + chunk_size, height)
                    x_start = x_chunk * chunk_size
                    x_end = min(x_start + chunk_size, width)
                    
                    # Show progress for large processing jobs
                    if total_chunks > 4 and (chunk_count % 20 == 0 or chunk_count == total_chunks):
                        print(f"  Processing chunk {chunk_count}/{total_chunks}")
                    
                    # Read chunk
                    window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    ndvi_chunk = src.read(1, window=window)
                    
                    # Create mask for valid data
                    if nodata is not None:
                        valid_mask = (ndvi_chunk != nodata)
                    else:
                        # If no nodata value is defined, check for NaN
                        valid_mask = ~np.isnan(ndvi_chunk)
                    
                    # Initialize C-factor chunk with NaN values
                    cfactor_chunk = np.full_like(ndvi_chunk, np.nan, dtype=np.float32)
                    
                    # Calculate C-factor for valid NDVI values using the formula
                    # C = e^((-alpha * NDVI)/(beta - NDVI))
                    
                    # Create a calculation mask that excludes invalid NDVI values
                    calc_mask = valid_mask & (ndvi_chunk < beta)
                    
                    if np.any(calc_mask):
                        # Apply the formula to valid pixels
                        numerator = -alpha * ndvi_chunk[calc_mask]
                        denominator = beta - ndvi_chunk[calc_mask]
                        
                        # Calculate the C-factor
                        cfactor_chunk[calc_mask] = np.exp(numerator / denominator)
                    
                    # Handle special case where NDVI ≥ beta
                    # In these cases, set C-factor to minimum value (high vegetation cover, minimal erosion)
                    # Using 0.01 as a minimum value as C-factor typically ranges from 0 to 1
                    cfactor_chunk[valid_mask & (ndvi_chunk >= beta)] = 0.01
                    
                    # Ensure C-factor is in valid range [0,1]
                    cfactor_chunk[cfactor_chunk > 1] = 1.0
                    cfactor_chunk[cfactor_chunk < 0] = 0.0
                    
                    # Collect statistics from valid results in this chunk
                    valid_cfactor_chunk = cfactor_chunk[~np.isnan(cfactor_chunk)]
                    if len(valid_cfactor_chunk) > 0:
                        # For memory efficiency, sample statistics rather than storing all values
                        if len(valid_cfactor_chunk) > 10000:
                            # Sample 10000 random values for statistics
                            sample_indices = np.random.choice(len(valid_cfactor_chunk), 10000, replace=False)
                            valid_cfactor_values.extend(valid_cfactor_chunk[sample_indices])
                        else:
                            valid_cfactor_values.extend(valid_cfactor_chunk)
                        
                        total_valid_pixels += len(valid_cfactor_chunk)
                    
                    # Write chunk to output
                    dst.write(cfactor_chunk.astype(np.float32), 1, window=window)
        
        # Calculate and display final statistics
        if len(valid_cfactor_values) > 0:
            valid_cfactor_array = np.array(valid_cfactor_values)
            print(f"\nC-factor calculation complete!")
            print(f"Statistics (based on {len(valid_cfactor_array):,} sampled pixels out of {total_valid_pixels:,} total valid pixels):")
            print(f"  - Min: {np.min(valid_cfactor_array):.6f}")
            print(f"  - Max: {np.max(valid_cfactor_array):.6f}")
            print(f"  - Mean: {np.mean(valid_cfactor_array):.6f}")
            print(f"  - Standard deviation: {np.std(valid_cfactor_array):.6f}")
            print(f"  - Median: {np.median(valid_cfactor_array):.6f}")
        else:
            print("Warning: No valid NDVI values found for C-factor calculation!")
        
        print(f"Output saved to: {cfactor_path}")
    
    return cfactor_path


# # Example usage:
# if __name__ == "__main__":
#     # Example paths - replace with your actual file paths
#     ndvi_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/NDVI_AOI_UTM.tif"
#     cfactor_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/Cfactor_AOI_UTM.tif"
    
#     calculate_cfactor_from_ndvi(ndvi_path, cfactor_path, chunk_size=4096)

# ============================================================================================================================
# Step 6 : Calculate Erosion Rates
# ===========================================================================================================================
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling

def calculate_erosion_rate(r_factor_path, k_factor_path, ls_factor_path, c_factor_path, 
                          output_path,  p_factor, resample_method='bilinear', chunk_size=1024, temp_dir = None):
    """
    Calculate soil erosion rate using the rusle model by integrating all erosion factors.
    Using rasterio only to avoid GDAL array dependency issues.
    
    rusle Equation: A = R × K × LS × C × P
    
    Parameters:
    -----------
    r_factor_path : str
        Path to the rainfall erosivity factor (R) raster file
    k_factor_path : str
        Path to the soil erodibility factor (K) raster file in US customary units
    ls_factor_path : str
        Path to the slope length and steepness factor (LS) raster file
    c_factor_path : str
        Path to the crop management factor (C) raster file
    output_path : str
        Path to save the resulting erosion rate (A) raster file
    p_factor : float, optional
        Conservation practice factor (P), default is 1.0 (no conservation practices)
    resample_method : str, optional
        Method used for resampling: 'bilinear', 'nearest', 'cubic', etc.
    chunk_size : int, optional
        Size of chunks for memory-efficient processing. Default is 1024 pixels.
        
    Returns:
    --------
    str
        Path to the output erosion rate raster
    """
    print("Starting rusle erosion rate calculation (Rasterio-only version with chunking)...")
    print(f"Using chunk size: {chunk_size}x{chunk_size} pixels")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Identify the reference raster (using R-factor as reference)
    print("Reading reference raster (R-factor) to establish dimensions...")
    with rasterio.open(r_factor_path) as src:
        reference_profile = src.profile.copy()
        reference_transform = src.transform
        reference_crs = src.crs
        reference_width = src.width
        reference_height = src.height
        reference_bounds = src.bounds
        
        print(f"Reference raster properties:")
        print(f"  - Resolution: {src.res}")
        print(f"  - CRS: {src.crs}")
        print(f"  - Dimensions: {src.width} x {src.height}")
        
        # Calculate number of chunks needed
        n_chunks_y = (reference_height + chunk_size - 1) // chunk_size
        n_chunks_x = (reference_width + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_y * n_chunks_x
        
        if total_chunks > 1:
            print(f"Large raster detected, will process {total_chunks} chunks ({n_chunks_y}x{n_chunks_x})")
        
        # Sample R-factor data for diagnostics
        sample_window = rasterio.windows.Window(0, 0, min(chunk_size, reference_width), min(chunk_size, reference_height))
        r_sample = src.read(1, window=sample_window)
        r_nodata = src.nodata
        
        if r_nodata is not None:
            r_valid_sample = (r_sample != r_nodata) & ~np.isnan(r_sample)
        else:
            r_valid_sample = ~np.isnan(r_sample)
        
        print(f"  - Sample R-factor validity: {np.sum(r_valid_sample)}/{r_sample.size} pixels")
        
        if np.sum(r_valid_sample) == 0:
            print("WARNING: No valid data found in R-factor sample")
    
    # Step 2: Create temporary directory for resampled rasters
    # Use provided temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    else:
        cleanup_temp = False
    try:
        # Step 3: Process and resample all factor rasters
        print("Processing and resampling factor rasters...")
        
        # Process K-factor
        print("Processing K-factor and converting to SI units...")
        k_factor_resampled_path = os.path.join(temp_dir, "rusle_k_factor_resampled.tif")
        k_factor_si_path = os.path.join(temp_dir, "rusle_k_factor_si.tif")
        
        # First process K-factor (without conversion)
        k_data = process_factor_raster(
            k_factor_path, k_factor_resampled_path, reference_profile, 
            reference_transform, reference_crs, reference_width, reference_height,
            resample_method, "K-factor", chunk_size
        )
        
        # Then apply K-factor conversion and save to SI file
        k_conversion_factor = 0.1317
        k_data_si = np.full_like(k_data, np.nan, dtype=np.float32)
        k_valid_mask = np.isfinite(k_data)
        
        if np.any(k_valid_mask):
            k_data_si[k_valid_mask] = k_data[k_valid_mask] * k_conversion_factor
            print(f"K-factor converted to SI units (factor: {k_conversion_factor})")
            print(f"  - Original range: {np.min(k_data[k_valid_mask]):.6f} to {np.max(k_data[k_valid_mask]):.6f}")
            print(f"  - SI range: {np.min(k_data_si[k_valid_mask]):.6f} to {np.max(k_data_si[k_valid_mask]):.6f}")
        else:
            print("WARNING: No valid K-factor data to convert!")
        
        # Save K-factor SI to file
        k_si_profile = reference_profile.copy()
        k_si_profile.update({'dtype': 'float32', 'nodata': np.nan})
        
        with rasterio.open(k_factor_si_path, 'w', **k_si_profile) as dst:
            dst.write(k_data_si.astype(np.float32), 1)
        
        print(f"K-factor SI saved to: {k_factor_si_path}")
        
        # Process LS-factor
        print("Processing LS-factor...")
        ls_factor_resampled_path = os.path.join(temp_dir, "rusle_ls_factor_resampled.tif")
        
        ls_data = process_factor_raster(
            ls_factor_path, ls_factor_resampled_path, reference_profile,
            reference_transform, reference_crs, reference_width, reference_height,
            resample_method, "LS-factor", chunk_size
        )
        
        # Process C-factor
        print("Processing C-factor...")
        c_factor_resampled_path = os.path.join(temp_dir, "rusle_c_factor_resampled.tif")
        
        c_data = process_factor_raster(
            c_factor_path, c_factor_resampled_path, reference_profile,
            reference_transform, reference_crs, reference_width, reference_height,
            resample_method, "C-factor", chunk_size
        )
        
        # Step 4: Calculate erosion rate using chunked processing
        print("Calculating erosion rate using rusle equation with chunking: A = R × K × LS × C × P...")
        
        # Prepare output profile
        output_profile = reference_profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'nodata': np.nan,
            'driver': 'GTiff'
        })
        
        # Initialize statistics collection
        valid_erosion_values = []
        total_valid_pixels = 0
        total_pixels = 0
        
        # Process erosion calculation in chunks
        with rasterio.open(r_factor_path) as r_src, \
            rasterio.open(k_factor_si_path) as k_src, \
            rasterio.open(ls_factor_resampled_path) as ls_src, \
            rasterio.open(c_factor_resampled_path) as c_src, \
            rasterio.open(output_path, 'w', **output_profile) as dst:
            
            chunk_count = 0
            
            for y_chunk in range(n_chunks_y):
                for x_chunk in range(n_chunks_x):
                    chunk_count += 1
                    
                    # Calculate chunk boundaries
                    y_start = y_chunk * chunk_size
                    y_end = min(y_start + chunk_size, reference_height)
                    x_start = x_chunk * chunk_size
                    x_end = min(x_start + chunk_size, reference_width)
                    
                    # Show progress for large processing jobs
                    if total_chunks > 4 and (chunk_count % 20 == 0 or chunk_count == total_chunks):
                        print(f"  Processing chunk {chunk_count}/{total_chunks}")
                    
                    # Read chunk from each factor
                    window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    
                    r_chunk = r_src.read(1, window=window)
                    k_chunk = k_src.read(1, window=window)
                    ls_chunk = ls_src.read(1, window=window)
                    c_chunk = c_src.read(1, window=window)
                    
                    # Create masks for valid data
                    r_nodata = r_src.nodata
                    if r_nodata is not None:
                        r_valid = (r_chunk != r_nodata) & np.isfinite(r_chunk)
                    else:
                        r_valid = np.isfinite(r_chunk)
                    
                    k_valid = np.isfinite(k_chunk)
                    ls_valid = np.isfinite(ls_chunk)
                    c_valid = np.isfinite(c_chunk)
                    
                    # Combined validity mask
                    valid_mask = r_valid & k_valid & ls_valid & c_valid
                    
                    # Initialize erosion chunk with NaN
                    erosion_chunk = np.full_like(r_chunk, np.nan, dtype=np.float32)
                    
                    # Calculate erosion for valid pixels
                    if np.any(valid_mask):
                        erosion_chunk[valid_mask] = (
                            r_chunk[valid_mask] * 
                            k_chunk[valid_mask] * 
                            ls_chunk[valid_mask] * 
                            c_chunk[valid_mask] * 
                            p_factor
                        )
                        
                        # Collect statistics from valid results in this chunk
                        valid_erosion_chunk = erosion_chunk[valid_mask]
                        
                        # For memory efficiency, sample statistics rather than storing all values
                        if len(valid_erosion_chunk) > 5000:
                            # Sample 5000 random values for statistics
                            sample_indices = np.random.choice(len(valid_erosion_chunk), 5000, replace=False)
                            valid_erosion_values.extend(valid_erosion_chunk[sample_indices])
                        else:
                            valid_erosion_values.extend(valid_erosion_chunk)
                        
                        total_valid_pixels += len(valid_erosion_chunk)
                    
                    total_pixels += erosion_chunk.size
                    
                    # Write chunk to output
                    dst.write(erosion_chunk.astype(np.float32), 1, window=window)
        
        # Calculate and display final statistics
        if len(valid_erosion_values) > 0:
            valid_erosion_array = np.array(valid_erosion_values)
            
            print(f"\nrusle erosion rate calculation complete!")
            print(f"Statistics (based on {len(valid_erosion_array):,} sampled pixels out of {total_valid_pixels:,} total valid pixels):")
            print(f"Valid pixel coverage: {total_valid_pixels}/{total_pixels} ({total_valid_pixels/total_pixels*100:.2f}%)")
            print(f"Erosion rate statistics (t/ha/yr):")
            print(f"  - Min: {np.min(valid_erosion_array):.6f}")
            print(f"  - Max: {np.max(valid_erosion_array):.6f}")
            print(f"  - Mean: {np.mean(valid_erosion_array):.6f}")
            print(f"  - Median: {np.median(valid_erosion_array):.6f}")
            print(f"  - Standard deviation: {np.std(valid_erosion_array):.6f}")
            
            # Calculate area exceeding tolerance levels
            tolerance = 5.0  # Example tolerance level in t/ha/yr
            percent_exceeding = (np.sum(valid_erosion_array > tolerance) / len(valid_erosion_array)) * 100
            print(f"  - Percent area exceeding {tolerance} t/ha/yr: {percent_exceeding:.2f}%")
        else:
            print("WARNING: No valid erosion values could be calculated!")
            print("This indicates that there is no spatial overlap where all four factors have valid data.")
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        try:
            for temp_file in [k_factor_resampled_path, k_factor_si_path, 
                            ls_factor_resampled_path, c_factor_resampled_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove all temporary files: {e}")
        
        print(f"Output saved to: {output_path}")
        return output_path
    finally:
        # Only clean up if we created the temp directory
        if cleanup_temp:
            shutil.rmtree(temp_dir)

def process_factor_raster(factor_path, output_path, reference_profile, reference_transform, 
                         reference_crs, reference_width, reference_height, resample_method, 
                         factor_name, chunk_size):
    """
    Process and resample a factor raster to match reference dimensions with chunking.
    
    Parameters:
    -----------
    factor_path : str
        Path to the input factor raster
    output_path : str
        Path to save the resampled factor raster
    reference_profile : dict
        Reference raster profile
    reference_transform : Affine
        Reference raster transform
    reference_crs : CRS
        Reference raster CRS
    reference_width : int
        Reference raster width
    reference_height : int
        Reference raster height
    resample_method : str
        Resampling method
    factor_name : str
        Name of the factor (for logging)
    chunk_size : int
        Chunk size for processing
        
    Returns:
    --------
    numpy.ndarray
        Processed factor data array
    """
    with rasterio.open(factor_path) as src:
        original_data = src.read(1)
        factor_nodata = src.nodata
        factor_transform = src.transform
        factor_crs = src.crs
        factor_width = src.width
        factor_height = src.height
        
        print(f"{factor_name} original properties:")
        print(f"  - Data type: {src.dtypes[0]}")
        print(f"  - NoData value: {factor_nodata}")
        print(f"  - Dimensions: {factor_width} x {factor_height}")
        
        # Check for valid values
        if factor_nodata is not None:
            valid_mask_orig = (original_data != factor_nodata) & np.isfinite(original_data)
        else:
            valid_mask_orig = np.isfinite(original_data)
        
        valid_count = np.sum(valid_mask_orig)
        if valid_count > 0:
            print(f"  - Valid {factor_name} pixels: {valid_count} ({valid_count/original_data.size*100:.2f}%)")
            print(f"  - {factor_name} range: {np.min(original_data[valid_mask_orig]):.6f} to {np.max(original_data[valid_mask_orig]):.6f}")
        else:
            print(f"  - WARNING: {factor_name} raster contains no valid data!")
        
        # Determine if resampling is needed
        needs_resampling = (factor_width != reference_width or 
                           factor_height != reference_height or 
                           factor_crs != reference_crs)
        
        if needs_resampling:
            print(f"Resampling {factor_name} to match reference raster...")
            
            # Initialize output array for resampled data
            resampled_data = np.full((reference_height, reference_width), np.nan, dtype=np.float32)
            
            # Set resampling method
            if resample_method.lower() == 'bilinear':
                resampling = Resampling.bilinear
            elif resample_method.lower() == 'nearest':
                resampling = Resampling.nearest
            elif resample_method.lower() == 'cubic':
                resampling = Resampling.cubic
            else:
                resampling = Resampling.bilinear
            
            # Perform reprojection
            try:
                reproject(
                    source=original_data,
                    destination=resampled_data,
                    src_transform=factor_transform,
                    src_crs=factor_crs,
                    dst_transform=reference_transform,
                    dst_crs=reference_crs,
                    resampling=resampling,
                    src_nodata=factor_nodata,
                    dst_nodata=np.nan
                )
                
                # Check resampled data
                valid_mask = np.isfinite(resampled_data)
                valid_count = np.sum(valid_mask)
                
                if valid_count > 0:
                    print(f"  - Valid {factor_name} pixels (after resampling): {valid_count} ({valid_count/resampled_data.size*100:.2f}%)")
                    print(f"  - {factor_name} range after resampling: {np.min(resampled_data[valid_mask]):.6f} to {np.max(resampled_data[valid_mask]):.6f}")
                else:
                    print(f"  - WARNING: {factor_name} raster contains no valid data after resampling!")
                
                factor_data = resampled_data
                
            except Exception as e:
                print(f"Error during {factor_name} resampling: {e}")
                print(f"Using original {factor_name} data without resampling")
                factor_data = original_data
        else:
            print(f"{factor_name} already matches reference dimensions, no resampling needed")
            factor_data = original_data
        
        # Save the processed factor to file
        factor_profile = reference_profile.copy()
        factor_profile.update({
            'dtype': 'float32',
            'nodata': np.nan
        })
        
        with rasterio.open(output_path, 'w', **factor_profile) as dst:
            dst.write(factor_data.astype(np.float32), 1)
        
        print(f"Processed {factor_name} saved to: {output_path}")
        
        return factor_data
# # Example usage:
# if __name__ == "__main__":
#     # Example paths - replace with your actual file paths
#     r_factor_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/Rfactor_UTM.tif"
#     k_factor_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/Kffactor_UTM.tif"
#     ls_factor_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/LS_factor_UTM.tif"
#     c_factor_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/Cfactor_AOI_UTM.tif"
#     output_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/user_data/1_outputs/Tallulah River/hydrogeo_vuln/rusle/RUSLE_erosion_rates_t_ha_yr_AOI_UTM.tif"
    
#     # Default chunking
#     calculate_erosion_rate(
#         r_factor_path, 
#         k_factor_path, 
#         ls_factor_path, 
#         c_factor_path, 
#         output_path, 
#         p_factor=1.0,
#         resample_method='bilinear',
#         chunk_size=4096
#     )
    
#     # Or with custom chunk size for memory-constrained systems
#     # calculate_erosion_rate(
#     #     r_factor_path, 
#     #     k_factor_path, 
#     #     ls_factor_path, 
#     #     c_factor_path, 
#     #     output_path, 
#     #     p_factor=1.0,
#     #     resample_method='bilinear',
#     #     chunk_size=4096
#     # )

# ================================================================================================================================
# Step 7 : Categorizing Erosion rasters in 1 to 5 scale for each watershed
# ====================================================================================================================================
def calculate_watershed_erosion_summary(watershed_path, erosion_raster_path, output_path):
    """
    Calculate the average of rusle erosion values within each watershed polygon,
    multiply by watershed area, assign rankings, and categorize watersheds based on erosion severity.
    Adds erosion, score, and category columns to the output.
    
    Parameters:
    -----------
    watershed_path : str
        Path to the watershed polygon shapefile
    erosion_raster_path : str
        Path to the rusle erosion raster (.tif) file
    output_path : str
        Path to save the resulting watershed shapefile with erosion summary data
        
    Returns:
    --------
    str
        Path to the output watershed shapefile
    """
    print(f"Processing watershed erosion summary...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Read the watershed polygons
    print("Reading watershed polygons...")
    watersheds = gpd.read_file(watershed_path)
    
    # Step 2: Read the erosion raster
    print("Reading erosion raster...")
    with rasterio.open(erosion_raster_path) as src:
        erosion_crs = src.crs
        erosion_res = src.res
        
        # Get pixel area in hectares (assuming projection is in meters)
        pixel_area_ha = (erosion_res[0] * erosion_res[1]) / 10000  # convert m² to ha
        
        # Check if CRS of watersheds matches erosion raster
        if watersheds.crs != erosion_crs:
            print(f"Reprojecting watersheds to match erosion raster CRS")
            watersheds = watersheds.to_crs(erosion_crs)
        
        # Step 3: Check for duplications using Point_ID if it exists
        if 'Point_ID' in watersheds.columns:
            # Check for duplicates in the ID field
            duplicate_count = watersheds['Point_ID'].duplicated().sum()
            if duplicate_count > 0:
                print(f"Found {duplicate_count} duplicated Point_ID values, dissolving to merge duplicates")
                watersheds = watersheds.dissolve(by='Point_ID', as_index=False)
        
        # Step 4: Calculate erosion metrics for each watershed
        print("Calculating erosion metrics for each watershed...")
        erosion_values = []
        watershed_ids = []
        
        total_watersheds = len(watersheds)
        for idx, watershed in enumerate(watersheds.itertuples()):
            if idx % max(1, int(total_watersheds/10)) == 0:  # Progress update every ~10%
                print(f"  Processing watershed {idx+1}/{total_watersheds}...")
            
            try:
                # Get the watershed geometry and create a mask
                geom = watershed.geometry
                if geom.is_empty:
                    print(f"  Warning: Empty geometry for watershed at index {idx}")
                    erosion_values.append(np.nan)
                    watershed_ids.append(getattr(watershed, 'Point_ID', idx))
                    continue
                
                # Fix any invalid geometries
                if not geom.is_valid:
                    geom = geom.buffer(0)
                    if not geom.is_valid:
                        print(f"  Warning: Could not fix invalid geometry for watershed at index {idx}")
                        erosion_values.append(np.nan)
                        watershed_ids.append(getattr(watershed, 'Point_ID', idx))
                        continue
                
                # Create a mask from the watershed geometry
                geom_json = [mapping(geom)]
                
                # Get the erosion data for this watershed
                # Use higher precision (float64) for calculations
                masked_data, masked_transform = mask(src, geom_json, crop=True, all_touched=False, nodata=np.nan)
                masked_data = masked_data.astype(np.float64)  # Use higher precision
                
                # Count valid (non-NaN) pixels
                valid_pixel_count = np.sum(~np.isnan(masked_data))
                
                # Calculate erosion metrics
                if valid_pixel_count > 0:
                    # Get average erosion rate (t/ha/yr)
                    avg_erosion_rate = np.nanmean(masked_data)
                    
                    # Get watershed area in hectares (use area_ha column if it exists)
                    if hasattr(watershed, 'area_ha'):
                        watershed_area_ha = getattr(watershed, 'area_ha')
                    else:
                        # Calculate area from geometry (convert to hectares)
                        watershed_area_ha = geom.area / 10000
                    
                    # Calculate total erosion (t/yr) = Average rate (t/ha/yr) × watershed area (ha)
                    erosion_value = avg_erosion_rate * watershed_area_ha
                else:
                    print(f"  Warning: No valid erosion data within watershed at index {idx}")
                    erosion_value = np.nan
                
                erosion_values.append(erosion_value)
                watershed_ids.append(getattr(watershed, 'Point_ID', idx))
                
            except Exception as e:
                print(f"  Error processing watershed at index {idx}: {e}")
                erosion_values.append(np.nan)
                watershed_ids.append(getattr(watershed, 'Point_ID', idx))
    
    # Step 5: Add the erosion value to the watershed geodataframe
    watersheds['erosion'] = erosion_values
    
    # Step 6: Apply rank-based scaling to assign scores from 0 to 5
    print("Applying rank-based scaling for classification...")
    
    # Function for ranking-based scaling to 0-5 range
    def rank_based_scale(values, min_score=0.0, max_score=5.0):
        """
        Apply a rank-based scaling approach to spread values evenly
        across the range from min_score to max_score.
        """
        values_array = np.array(values)
        valid_mask = ~np.isnan(values_array)
        
        if not np.any(valid_mask):
            return values_array  # Return as is if all are NaN
            
        if np.all(values_array[valid_mask] == values_array[valid_mask][0]):
            # If all values are the same, assign the middle score
            result = np.full_like(values_array, (min_score + max_score) / 2)
            result[~valid_mask] = np.nan
            return result
            
        valid_values = values_array[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Create a ranking (sorting) of the values
        rank_order = np.argsort(valid_values)
        
        # Create evenly spaced ranks from min_score to max_score
        n_valid = len(valid_values)
        if n_valid > 1:
            ranks = np.linspace(min_score, max_score, n_valid)
            
            # Assign ranks based on the sorted order
            scaled_values = np.zeros_like(valid_values)
            scaled_values[rank_order] = ranks
        else:
            # If only one value, assign middle score
            scaled_values = np.array([(min_score + max_score) / 2])
        
        # Put the result back with NaN values preserved
        result = np.full_like(values_array, np.nan)
        result[valid_indices] = scaled_values
        
        return result
    
    # Apply the rank-based scaling to the erosion values
    scores = rank_based_scale(watersheds['erosion'].values)
    watersheds['score'] = scores
    
    # Step 7: Categorize scores into vulnerability classes
    print("Categorizing watersheds based on erosion scores...")
    
    # Function to categorize scores
    def score_to_category(score):
        """
        Categorize standardized scores into vulnerability classes.
        0-1: Very Low
        1-2: Low
        2-3: Moderate
        3-4: High
        4-5: Very High
        """
        if np.isnan(score):
            return "No Data"
        
        # Using thresholds for a 0-5 scale
        if score < 1.0:
            return "Very Low"
        elif score < 2.0:
            return "Low"
        elif score < 3.0:
            return "Moderate"
        elif score < 4.0:
            return "High"
        else:  # 4.0 to 5.0
            return "Very High"
    
    # Apply categorization to the scores
    watersheds['category'] = watersheds['score'].apply(score_to_category)
    
    # Step 8: Generate summary of category distribution
    print("Category distribution:")
    category_counts = watersheds['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  - {category}: {count} watersheds ({count/len(watersheds)*100:.1f}%)")
    
    # Step 9: Remove old columns if they exist
    for col in ['erosion_sum_t_yr', 'sum_score', 'sum_category']:
        if col in watersheds.columns:
            watersheds = watersheds.drop(columns=[col])
    
    # Step 10: Save the result to a new shapefile
    print(f"Saving results to {output_path}...")
    watersheds.to_file(output_path)
    
    print("Watershed erosion summary processing complete.")
    return output_path   
   
# # Example usage:
# if __name__ == "__main__":
#     watershed_path = "/home/smdsgit/SouravDSGit/Culvert_web_app-main/instance/user_data/1_outputs/WS_deln/final_flag_removed_ws_polygon_filtered_by_area_UTM.shp"
#     erosion_raster_path = "/home/smdsgit/SouravDSGit/Culvert_web_app-main/instance/user_data/1_outputs/hydrogeo_vuln/rusle_erosion_rates_t_ha_yr_AOI_UTM.tif"
#     output_path = "/home/smdsgit/SouravDSGit/Culvert_web_app-main/instance/user_data/1_outputs/hydrogeo_vuln/rusle_watersheds_with_erosion.shp"
    
#     calculate_watershed_erosion_summary(watershed_path, erosion_raster_path, output_path)


# =====================================================================================================================
# Step 8 : Final Main Function
# ======================================================================================================================
import sys
from contextlib import redirect_stdout
def run_rusle_analysis(atlas14_dir_path, 
                    dem_path, 
                    output_100yr_30min_raster_path, 
                    categorized_100yr_30min_raster_path,
                    Rfactor_output_raster_path,
                    gssurgo_soil_data_directory_path,
                    usa_states_shapefile_path,
                    kffactor_raster_output_path,
                    flow_acc_path,
                    ls_factor_raster_output_path,
                    ndvi_input_path,
                    aoi_NDVI_raster_output_path,
                    cfactor_raster_output_path,
                    p_factor,
                    rusle_erosion_rate_raster_output_path,
                    watershed_polygon_shapefile_path,
                    categorized_risk_rusle_output_shapefile_path,
                    temp_file_path=None,
                    user_id=None,
                    project_name=None,
                    task_type=None,
                    check_cancellation_func=None):
    # Create centralized temp directory
    if temp_file_path is None:
        temp_dir = tempfile.mkdtemp(prefix="rusle_analysis_")
        cleanup_temp = True
    else:
        temp_dir = temp_file_path
        cleanup_temp = False
        # Ensure the temp directory exists
        os.makedirs(temp_dir, exist_ok=True)
    
    # Setup log file
    log_file_path = os.path.join(os.path.dirname(rusle_erosion_rate_raster_output_path), 'processing_log.txt')
    
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
        print("STARTING RUSLE ANALYSIS WORKFLOW")
        print("=" * 80)
        print(f"Using centralized temporary directory: {temp_dir}")
        print("Starting RUSLE analysis pipeline...")
        
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 1: Processing NOAA Atlas14 data ===")
        process_noaa_atlas14_for_boundary(atlas14_dir_path, 
                                          watershed_polygon_shapefile_path, 
                                          dem_path, 
                                          output_100yr_30min_raster_path, 
                                          categorized_100yr_30min_raster_path,
                                          temp_dir=temp_dir,method_prefix="rusle")
        
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 2: Calculating R-factor ===")
        calculate_r_factor(output_100yr_30min_raster_path, Rfactor_output_raster_path)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 3: Extracting K-factor from gSSURGO ===")
        gssurgo_to_kffactor_raster(gssurgo_soil_data_directory_path,
                                    watershed_polygon_shapefile_path,
                                    dem_path,
                                    usa_states_shapefile_path,
                                    kffactor_raster_output_path,
                                    temp_dir=temp_dir)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 4: Calculating LS-factor ===")
        calculate_ls_factor(dem_path, flow_acc_path, ls_factor_raster_output_path, 
                           watershed_polygon_shapefile_path, chunk_size=4096, temp_dir=temp_dir)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 5: Processing NDVI data ===")
        process_ndvi_raster(ndvi_input_path, dem_path, 
                            watershed_polygon_shapefile_path, 
                            aoi_NDVI_raster_output_path, 
                            ndvi_scale_factor=10000.0,
                            buffer_dist=500.0,
                            chunk_size=4096,
                            temp_dir=temp_dir)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 6: Calculating C-factor ===")
        calculate_cfactor_from_ndvi(aoi_NDVI_raster_output_path, cfactor_raster_output_path, 
                                   alpha=2, beta=1, chunk_size=4096)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 7: Calculating erosion rates ===")
        calculate_erosion_rate(Rfactor_output_raster_path, 
                               kffactor_raster_output_path, 
                               ls_factor_raster_output_path, 
                               cfactor_raster_output_path, 
                               rusle_erosion_rate_raster_output_path, 
                               p_factor=p_factor, 
                               resample_method='bilinear',
                               chunk_size=4096,
                               temp_dir= temp_dir)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n=== Step 8: Calculating watershed erosion summary ===")
        calculate_watershed_erosion_summary(watershed_polygon_shapefile_path, 
                                            rusle_erosion_rate_raster_output_path, 
                                            categorized_risk_rusle_output_shapefile_path)
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
            print(f"RUSLE analysis failed: {e}")
        
        print("\n🎉 RUSLE analysis pipeline completed successfully!")
        return True
        
    except TaskCancelledError:
        print(f"\n❌ RUSLE analysis was cancelled by user")
        raise  # RE-RAISE THE CANCELLATION EXCEPTION
        
    except Exception as e:
        print(f"\n❌ RUSLE analysis failed at step: {str(e)}")
        return False
        
    finally:
        # Restore stdout and close log
        sys.stdout = dual_output.terminal
        dual_output.close()
        # Clean up only if we created the temp directory
        if cleanup_temp and os.path.exists(temp_dir):
            print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("✓ Temporary directory cleaned up successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temporary directory: {e}")