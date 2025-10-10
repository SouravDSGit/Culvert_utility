"""
Optimized Water Erosion Prediction Project (WEPP) implementation.
This module follows the WEPP User Summary (NSERL Report NO. 11, July 1995) specifications
and provides an efficient, parallelized implementation for watershed analysis.

The module handles processing of:
- Digital Elevation Models (DEM) for slope data
- GSSURGO soil data
- NLCD land cover data for management parameters
- Climate data using Daymet or CLIGEN 5.322
- WEPP model execution and result parsing

Author: Revised implementation following WEPP User Summary guidelines
"""

import os
import subprocess
import tempfile
import threading
import logging
import traceback
import shutil
import hashlib
import json
import time
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from urllib.parse import quote

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import aiohttp
import psutil
import requests
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from shapely.geometry import mapping, shape, Point, MultiPoint
from shapely.prepared import prep
from rtree import index
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-local storage for thread safety
thread_local = threading.local()

# Global cache containers
CLIMATE_CACHE = {}

# ================================================
# UTILITY FUNCTIONS
# ================================================

def get_cache_dir():
    """Create and return a persistent cache directory for climate data."""
    cache_dir = os.path.expanduser("~/.wepp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_cache_key(lat, lon, start_date, end_date):
    """Generate a unique cache key for climate data."""
    key_string = f"{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
    return hashlib.md5(key_string.encode()).hexdigest()

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

def get_thread_temp_dir(base_temp_dir=None):
    """Get thread-local temporary directory."""
    if not hasattr(thread_local, 'temp_dir'):
        thread_id = threading.get_ident()
        thread_local.temp_dir = tempfile.mkdtemp(
            dir=base_temp_dir,
            prefix=f"wepp_thread_{thread_id}_"
        )
    return thread_local.temp_dir

def cleanup_temp_dirs():
    """Clean up all thread-local temporary directories."""
    try:
        for thread_id, thread in threading._active.items():
            if hasattr(thread, 'thread_local') and hasattr(thread.thread_local, 'temp_dir'):
                temp_dir = thread.thread_local.temp_dir
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Error cleaning up temp directories: {e}")

def determine_optimal_workers(override=None):
    """Determine optimal number of worker threads based on system resources."""
    if override is not None:
        return min(override, os.cpu_count() or 1)
    
    cpu_count = os.cpu_count() or 1
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Adaptive worker count based on system resources
    if memory_gb > 32:
        return min(cpu_count, 16)  # High-end system
    elif memory_gb > 16:
        return min(cpu_count, 8)   # Medium system
    elif memory_gb > 8:
        return min(cpu_count // 2, 4)  # Lower-end system
    else:
        return 2  # Very limited resources

# ================================================
# SOIL DATA PROCESSING
# ================================================

def create_soil_file(hillslope, soils, output_dir, hillslope_id=None):
    """
    Create a WEPP soil file for a hillslope using GSSURGO data.
    
    Args:
        hillslope (GeoSeries): Hillslope geometry
        soils (GeoDataFrame): Soil data
        output_dir (str): Directory to save output
        hillslope_id (str, optional): Hillslope ID
        
    Returns:
        str: Path to WEPP soil file
    """
    # Simplify ID assignment
    if hillslope_id is None:
        hillslope_id = str(hillslope['id']) if 'id' in hillslope else f"hillslope_{hillslope.name}"
    
    soil_file = os.path.join(output_dir, f"soil_{hillslope_id}.sol")
    
    # Find soil that intersects with hillslope
    intersecting_soils = soils[soils.intersects(hillslope.geometry)]
    
    if intersecting_soils.empty:
        # Use nearest soil if no intersection
        nearest_idx = soils.distance(hillslope.geometry.centroid).idxmin()
        soil_props = soils.loc[nearest_idx]
    else:
        # Find dominant soil by intersection area
        intersecting_soils['intersection'] = intersecting_soils.geometry.intersection(hillslope.geometry).area
        soil_props = intersecting_soils.sort_values(by='intersection', ascending=False).iloc[0]
    
    # Get key properties with defaults
    kffact = 0.20  # Default value
    if 'kffact' in soil_props and not pd.isna(soil_props['kffact']):
        kffact = float(soil_props['kffact'])
    
    soil_texture = "loam"  # Default texture
    if 'texdesc' in soil_props and not pd.isna(soil_props['texdesc']):
        soil_texture = soil_props['texdesc']
    
    # Calculate parameters according to WEPP manual formulas
    ki = kffact * 2728000  
    kr = 0.00197 + 0.00030 * 15 + 0.03863 * np.exp(-1.84 * 2.0)
    critical_shear = 2.67 + 0.065 * 20 - 0.058 * 15
    
    # Get clay content with default
    clay_content = 20  # Default value
    if 'claytotal_r' in soil_props and not pd.isna(soil_props['claytotal_r']):
        clay_content = float(soil_props['claytotal_r'])
    
    # Hydraulic conductivity calculation
    if clay_content <= 40:
        effective_conductivity = -0.265 + 0.0086 * 15**1.8 + 11.46 * 10**-0.75
    else:
        effective_conductivity = 0.0066 * np.exp(244/clay_content)
    
    # Write WEPP soil file
    with open(soil_file, 'w') as f:
        f.write("95.7\n")
        f.write(f"# WEPP soil file for hillslope {hillslope_id} - {soil_texture}\n")
        f.write("1 1\n")
        f.write(f"'{soil_texture.upper()}' '{soil_texture}' 3 0.14 0.75 {ki:.5f} {kr:.5f} {critical_shear:.2f} {effective_conductivity:.2f}\n")
        f.write(f"200 40 {clay_content:.1f} 2.0 10.0 5.0\n")
        f.write(f"400 35 {clay_content+5:.1f} 1.0 8.0 10.0\n")
        f.write(f"1000 30 {clay_content+8:.1f} 0.5 6.0 15.0\n")
    
    return soil_file

def process_gssurgo_data(gssurgo_zip_path, boundary_shapefile, output_dir):
    """
    Extract relevant soil data from GSSURGO zipfile for WEPP processing.
    
    Args:
        gssurgo_zip_path (str): Path to GSSURGO zip file
        boundary_shapefile (str): Path to boundary shapefile
        output_dir (str): Directory to save output
        
    Returns:
        GeoDataFrame: Processed soil data with required WEPP parameters
    """
    # Check if processed soil file exists
    soil_output = os.path.join(output_dir, "processed_soil_data.gpkg")
    if os.path.exists(soil_output):
        logger.info(f"Using existing processed soil data: {soil_output}")
        return gpd.read_file(soil_output)
    
    logger.info(f"Processing GSSURGO data from: {gssurgo_zip_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Extracting GSSURGO to: {temp_dir}")
        
        # Unzip GSSURGO data
        import zipfile
        with zipfile.ZipFile(gssurgo_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find geodatabase
        gdb_path = None
        for root, dirs, _ in os.walk(temp_dir):
            gdb_dirs = [d for d in dirs if d.endswith('.gdb')]
            if gdb_dirs:
                gdb_path = os.path.join(root, gdb_dirs[0])
                break
        
        if not gdb_path:
            raise FileNotFoundError(f"No geodatabase found in GSSURGO zip: {gssurgo_zip_path}")
        
        logger.info(f"Found geodatabase: {gdb_path}")
        
        # Read key tables
        mupolygon = gpd.read_file(gdb_path, layer='MUPOLYGON')
        component = gpd.read_file(gdb_path, layer='component')
        chorizon = gpd.read_file(gdb_path, layer='chorizon')
        
        # Read texture data
        texture = None
        texture_field = None
        
        for layer_name in ['chtexturegrp', 'chtexture']:
            try:
                texture = gpd.read_file(gdb_path, layer=layer_name)
                for field in ['texdesc', 'texcl']:
                    if field in texture.columns:
                        texture_field = field
                        break
                if texture_field:
                    break
            except Exception:
                continue
        
        # Read mapunit table
        mapunit = gpd.read_file(gdb_path, layer='mapunit')
        
        # Process K factor from chorizon
        if 'kffact' in chorizon.columns:
            chorizon['kffact'] = pd.to_numeric(chorizon['kffact'], errors='coerce')
            chorizon = chorizon.dropna(subset=['kffact'])
        else:
            # Estimate K factor if not available
            logger.warning("kffact not found in chorizon, estimating from texture")
            if all(x in chorizon.columns for x in ['sandtotal_r', 'silttotal_r', 'claytotal_r']):
                chorizon['kffact'] = (chorizon['silttotal_r'] * 0.3 + chorizon['claytotal_r'] * 0.2) / 100
        
        # Join soil data with necessary columns
        needed_horizon_cols = ['cokey', 'kffact', 'chkey', 'claytotal_r', 'sandtotal_r', 'silttotal_r']
        needed_component_cols = ['cokey', 'mukey', 'comppct_r']
        
        horizon_component = pd.merge(
            chorizon[needed_horizon_cols],
            component[needed_component_cols],
            on='cokey',
            how='inner'
        )
        
        # Join with texture data if available
        if texture is not None and texture_field is not None:
            horizon_component = pd.merge(
                horizon_component,
                texture[['chkey', texture_field]],
                on='chkey',
                how='left'
            )
            horizon_component = horizon_component.rename(columns={texture_field: 'texdesc'})
        
        # Group by mukey and aggregate
        grouped = horizon_component.groupby('mukey').apply(lambda x: pd.Series({
            'kffact': np.average(x['kffact'], weights=x['comppct_r']),
            'claytotal_r': np.average(x['claytotal_r'], weights=x['comppct_r']) if 'claytotal_r' in x else None,
            'sandtotal_r': np.average(x['sandtotal_r'], weights=x['comppct_r']) if 'sandtotal_r' in x else None,
            'texdesc': x['texdesc'].mode().iloc[0] if 'texdesc' in x and not x['texdesc'].isna().all() else 'loam'
        })).reset_index()
        
        # Join with spatial data
        soil_gdf = pd.merge(
            mupolygon,
            grouped,
            left_on='MUKEY',
            right_on='mukey',
            how='inner'
        )
        
        # Convert to GeoDataFrame
        soil_gdf = gpd.GeoDataFrame(soil_gdf, geometry='geometry', crs=mupolygon.crs)
        
        # Process boundary and clip soil data
        boundary = gpd.read_file(boundary_shapefile)
        if soil_gdf.crs != boundary.crs:
            soil_gdf = soil_gdf.to_crs(boundary.crs)
        
        soil_gdf = gpd.clip(soil_gdf, boundary)
        
        # Save processed data
        soil_gdf.to_file(soil_output, driver="GPKG")
        
        logger.info(f"Processed soil data saved to: {soil_output}")
        return soil_gdf

# ================================================
# CLIMATE DATA PROCESSING
# ================================================

async def download_daymet_data(lat, lon, start_date, end_date, session):
    """
    Download Daymet climate data for a single coordinate.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        session (aiohttp.ClientSession): Async HTTP session
        
    Returns:
        pd.DataFrame: Daymet climate data
    """
    # Check memory cache
    cache_key = get_cache_key(lat, lon, start_date, end_date)
    if cache_key in CLIMATE_CACHE:
        return CLIMATE_CACHE[cache_key]
    
    # Check disk cache
    cache_dir = get_cache_dir()
    cache_file = os.path.join(cache_dir, f"daymet_{cache_key}.csv")
    
    if os.path.exists(cache_file):
        try:
            # Check cache age (valid for 30 days)
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 30 * 24 * 60 * 60:  # 30 days in seconds
                df = pd.read_csv(cache_file, parse_dates=['date'])
                df.set_index('date', inplace=True)
                CLIMATE_CACHE[cache_key] = df
                return df
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
    
    # Build Daymet API URL
    variables = ["tmax", "tmin", "prcp", "srad", "dayl", "vp", "swe"]
    vars_param = ",".join(variables)
    url = f"https://daymet.ornl.gov/single-pixel/api/data?lat={lat}&lon={lon}&vars={vars_param}&start={start_date}&end={end_date}"
    
    # Retry logic
    max_retries = 3
    retry_delay = 1  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download Daymet data: HTTP {response.status}")
                
                text = await response.text()
                
                # Validate response format
                if "year,yday" not in text:
                    raise ValueError("Invalid response format - not a valid Daymet CSV")
                
                # Find header line
                header_line = -1
                for i, line in enumerate(text.split('\n')):
                    if 'year,yday' in line:
                        header_line = i
                        break
                
                if header_line == -1:
                    raise ValueError("Could not find header line in response")
                
                # Parse CSV data
                csv_content = '\n'.join(text.split('\n')[header_line:])
                df = pd.read_csv(pd.StringIO(csv_content))
                
                # Process the dataframe
                for col in df.columns:
                    if col in ['year', 'yday']:
                        df[col] = df[col].astype(int)
                    else:
                        df[col] = df[col].astype(float)
                
                # Create date column
                df['date'] = df.apply(lambda row: pd.Timestamp(int(row['year']), 1, 1) +
                                    pd.Timedelta(days=int(row['yday'])-1), axis=1)
                
                # Set date as index
                df = df.sort_values('date')
                df.set_index('date', inplace=True)
                
                # Cache the result
                CLIMATE_CACHE[cache_key] = df
                
                # Save to disk cache
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_csv(cache_file)
                
                return df
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt+1} failed for {lat:.4f}, {lon:.4f}: {e}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts for {lat:.4f}, {lon:.4f}: {e}")
                raise RuntimeError(f"Failed to download Daymet data: {e}")
    
    # This should not be reached due to the exception above
    raise RuntimeError(f"Failed to download Daymet data after {max_retries} attempts")

def create_par_file_from_daymet(climate_data, output_file, station_name, coords):
    """
    Create a CLIGEN parameter (PAR) file from Daymet climate data.
    
    Args:
        climate_data (pd.DataFrame): Daymet climate data
        output_file (str): Path to output PAR file
        station_name (str): Station name
        coords (tuple): (lon, lat) coordinates
    """
    lon, lat = coords
    
    # Create monthly statistics
    climate_data = climate_data.copy()
    climate_data['month'] = climate_data.index.month
    
    # Calculate required monthly statistics for CLIGEN
    monthly_stats = climate_data.groupby('month').agg({
        'prcp': ['mean', 'std', lambda x: x.skew() if len(x) > 2 else 0, 
                lambda x: (x > 0).mean()],  # Precipitation stats
        'tmax': 'mean',  # Max temperature
        'tmin': 'mean',  # Min temperature
        'srad': 'mean'   # Solar radiation
    })
    
    # Flatten column names
    monthly_stats.columns = ['_'.join(col) if isinstance(col, tuple) else col 
                            for col in monthly_stats.columns.values]
    
    # Fill missing values
    monthly_stats = monthly_stats.fillna({
        'prcp_std': 0.1,
        'prcp_<lambda_0>': 0.0,  # skewness
        'prcp_<lambda_1>': 0.1   # wet day probability
    })
    
    # Write PAR file according to the format in the WEPP manual
    with open(output_file, 'w') as f:
        # Header line with station ID and coordinates
        f.write(f"{station_name:<20} {lat:.4f} {lon:.4f} 0.0\n")
        
        # Monthly precipitation means (mm)
        f.write("  MEAN P  ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'prcp_mean']:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Standard deviation of daily precipitation
        f.write("  S DEV P ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'prcp_std']:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Skewness of daily precipitation
        f.write("  SKEW P  ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'prcp_<lambda_0>']:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Probability of wet day following dry day
        f.write("  P(W|D)  ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'prcp_<lambda_1>']:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Probability of wet day following wet day (slightly higher than P(W|D))
        f.write("  P(W|W)  ")
        f.write(" ".join(f"{min(1.0, monthly_stats.loc[month, 'prcp_<lambda_1>'] * 1.5):6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Maximum temperature means (C)
        f.write("  TMAX AV ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'tmax_mean']:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Minimum temperature means (C)
        f.write("  TMIN AV ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'tmin_mean']:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Solar radiation (convert from W/m² to langleys/day)
        f.write("  SOL.RAD ")
        f.write(" ".join(f"{monthly_stats.loc[month, 'srad_mean'] * 2.064:6.2f}" for month in range(1, 13)))
        f.write("\n")
        
        # Add remaining parameters with default values
        # Using WEPP manual recommended defaults
        default_params = {
            "T.PEAK": [0.50] * 12,   # Time to peak rainfall intensity
            "R MX.5": [0.80] * 12,   # Max 30min/total ratio
            "MEAN W": [3.00] * 12,   # Mean wind speed
            "DIR.WND": [180.00] * 12, # Wind direction
            "SD WND": [1.00] * 12,   # Wind standard deviation
            "DEW PT": [10.00] * 12,  # Dew point temp
            "SD DEW": [2.00] * 12,   # Dew point std dev
            "% SUN": [50.00] * 12,   # % sunshine
            "SD SUN": [10.00] * 12,  # % sunshine std dev
            "MX.T SD": [5.00] * 12,  # Max temp std dev
            "MN.T SD": [5.00] * 12,  # Min temp std dev
            "MX-MN T": [10.00] * 12, # Max-min temp
            "TPK SD": [0.10] * 12,   # Time to peak std dev
            "R.5 SD": [0.10] * 12,   # 30min/total ratio std dev
            "RMX SD": [0.20] * 12,   # Max rainfall intensity std dev
            "SD DIR": [45.00] * 12,  # Wind direction std dev
            "CD TMAX": [0.60] * 12,  # Temp max autocorrelation
            "CD TMIN": [0.60] * 12,  # Temp min autocorrelation
            "CD DEWP": [0.60] * 12,  # Dew point autocorrelation
            "CD PRCP": [0.50] * 12,  # Precipitation autocorrelation
            "CD SOLR": [0.50] * 12,  # Solar rad autocorrelation
            "CD WIND": [0.50] * 12,  # Wind autocorrelation
            "CD WDIR": [0.50] * 12,  # Wind dir autocorrelation
            "Y.TMAX1": [0.00] * 12,  # Temp max harmonics
            "Y.TMAX2": [0.00] * 12,
            "Y.TMAX3": [0.00] * 12,
            "Y.TMIN1": [0.00] * 12,  # Temp min harmonics
            "Y.TMIN2": [0.00] * 12,
            "Y.TMIN3": [0.00] * 12,
            "Y.DEWP1": [0.00] * 12,  # Dew point harmonics
            "Y.DEWP2": [0.00] * 12,
            "Y.DEWP3": [0.00] * 12,
            "Y.PRCP1": [0.00] * 12,  # Precip harmonics
            "Y.PRCP2": [0.00] * 12,
            "Y.PRCP3": [0.00] * 12
        }
        
        # Write each parameter
        for param, values in default_params.items():
            f.write(f"  {param:7} ")
            f.write(" ".join(f"{v:6.2f}" for v in values))
            f.write("\n")

def update_cli_file(input_file, output_file, station_name, coords, climate_data):
    """
    Update CLI file with proper WEPP formatting.
    
    Args:
        input_file (str): Input CLI file generated by CLIGEN
        output_file (str): Output CLI file path
        station_name (str): Station name
        coords (tuple): (lon, lat) coordinates
        climate_data (pd.DataFrame): Climate data for reference
    """
    lon, lat = coords
    
    # Read the input CLI file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Write formatted output file according to WEPP manual format
    with open(output_file, 'w') as f:
        # Line 1: CLIGEN version number (4.0 for WEPP compatibility)
        f.write("4.0\n")
        
        # Line 2: Simulation mode (1=continuous), no breakpoint data, wind information exists
        f.write("1 0 0\n")
        
        # Line 3: Station information
        f.write(f"Station: {station_name} lat={lat:.4f} lon={lon:.4f}\n")
        
        # Line 4: Variable headers for climate parameters
        f.write("da mo year prcp dur tp ip tmax tmin rad w-vl w-dir tdew\n")
        
        # Line 5: Station coordinates and info
        years_of_data = climate_data.index.year.nunique()
        begin_year = pd.to_datetime(climate_data.index).year.min()
        f.write(f"{lat:.2f} {lon:.2f} 0.0 {years_of_data} {begin_year} {years_of_data}\n")
        
        # Copy monthly temperature, radiation, and precipitation statistics
        # Lines 6-14 from CLIGEN output
        for i in range(5, 14):
            if i < len(lines):
                f.write(lines[i])
            else:
                f.write("\n")  # Empty line if original is missing
        
        # Line 15: Daily variables and dimensions
        f.write("da mo year prcp dur tp ip tmax tmin rad w-vl w-dir tdew\n")
        f.write("-- -- ---- ---- ---- -- -- ---- ---- ---- ---- ----- ----\n")
        
        # Copy daily data from line 16 onward
        for i in range(15, len(lines)):
            f.write(lines[i])

def create_par_files_for_grid(climate_data, cligen_path):
    """
    Create PAR files for all grid points in the climate data.
    
    Args:
        climate_data (dict): Climate data from download_boundary_climate_data
        cligen_path (str): Path to CLIGEN executable
        
    Returns:
        dict: Updated climate data with PAR file paths
    """
    # Create PAR files for each grid point
    par_files = {}
    climate_dir = climate_data['output_dir']
    
    for grid_point, data in climate_data['grid_points'].items():
        lon, lat = grid_point
        grid_id = f"grid_{lat:.4f}_{lon:.4f}"
        par_file = os.path.join(climate_dir, f"par_{grid_id}.par")
        
        # Create the PAR file
        create_par_file_from_daymet(data, par_file, grid_id, grid_point)
        par_files[grid_point] = par_file
    
    # Add PAR files to climate data
    climate_data['par_files'] = par_files
    return climate_data

def find_nearest_grid_point(climate_data, lon, lat):
    """
    Find the nearest grid point to the given coordinates.
    
    Args:
        climate_data (dict): Climate data with spatial index
        lon (float): Longitude
        lat (float): Latitude
        
    Returns:
        tuple: (lon, lat) of nearest grid point
    """
    if 'spatial_index' not in climate_data:
        return None
    
    # Query spatial index for 5 nearest points
    idx = climate_data['spatial_index']
    nearest_idxs = list(idx.nearest((lon, lat, lon, lat), 5))
    
    if not nearest_idxs:
        return None
    
    # Find truly nearest point by calculating distance
    grid_points = list(climate_data['grid_points'].keys())
    min_dist = float('inf')
    nearest = None
    
    for i in nearest_idxs:
        if i >= len(grid_points):
            continue
            
        grid_point = grid_points[i]
        grid_lon, grid_lat = grid_point
        dist = ((grid_lon - lon) ** 2 + (grid_lat - lat) ** 2) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            nearest = grid_point
    
    return nearest

def create_climate_file_for_hillslope(climate_data, coords, hillslope_id, cligen_path, temp_dir=None):
    """
    Create a WEPP climate file for a hillslope using the boundary climate data.
    
    Args:
        climate_data (dict): Climate data from download_boundary_climate_data
        coords (tuple): (lon, lat) coordinates of hillslope
        hillslope_id (str): Hillslope ID
        cligen_path (str): Path to CLIGEN executable
        temp_dir (str, optional): Temporary directory
        
    Returns:
        str: Path to WEPP climate file
    """
    climate_dir = climate_data['output_dir']
    climate_file = os.path.join(climate_dir, f"climate_{hillslope_id}.cli")
    
    # Check if climate file already exists
    if os.path.exists(climate_file) and os.path.getsize(climate_file) > 0:
        return climate_file
    
    # Get thread-local temp directory
    if temp_dir is None:
        temp_dir = get_thread_temp_dir()
    
    # Extract coordinates
    lon, lat = coords
    
    # Find nearest grid point
    nearest_point = find_nearest_grid_point(climate_data, lon, lat)
    if nearest_point is None:
        raise ValueError(f"No grid points available for hillslope {hillslope_id}")
    
    # Get PAR file and climate data for nearest point
    if 'par_files' not in climate_data or nearest_point not in climate_data['par_files']:
        raise ValueError(f"No PAR file available for nearest grid point")
    
    par_file = climate_data['par_files'][nearest_point]
    point_climate_data = climate_data['grid_points'][nearest_point]
    
    # Calculate years for CLIGEN
    start_date = climate_data['start_date']
    end_date = climate_data['end_date']
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    num_years = end_year - start_year + 1
    
    # Run CLIGEN
    temp_cli_file = os.path.join(temp_dir, f"temp_{hillslope_id}.cli")
    cmd = [
        cligen_path,
        f"-i{par_file}",
        f"-o{temp_cli_file}",
        f"-b{start_year}",
        f"-y{num_years}",
        "-t5"  # Continuous simulation type
    ]
    
    # Ensure CLIGEN is executable
    if not os.access(cligen_path, os.X_OK):
        os.chmod(cligen_path, 0o755)
    
    # Run CLIGEN
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"CLIGEN stderr: {process.stderr}")
            raise RuntimeError(f"CLIGEN execution failed for hillslope {hillslope_id}")
        
        # Validate the generated file
        if not os.path.exists(temp_cli_file) or os.path.getsize(temp_cli_file) == 0:
            raise RuntimeError("CLIGEN generated an empty climate file")
        
        # Update the CLI file with proper formatting
        update_cli_file(temp_cli_file, climate_file, hillslope_id, coords, point_climate_data)
        
        return climate_file
        
    except Exception as e:
        logger.error(f"Error creating climate file for hillslope {hillslope_id}: {e}")
        raise RuntimeError(f"Failed to generate climate file for hillslope {hillslope_id}")

def process_watershed_climate(boundary_shapefile, start_date, end_date, output_dir, cligen_path, max_concurrent=10):
    """
    Process climate data for the entire study area boundary (not per watershed).
    
    Args:
        boundary_shapefile (str): Path to overall boundary shapefile
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Directory to save outputs
        cligen_path (str): Path to CLIGEN executable
        max_concurrent (int): Maximum number of concurrent downloads
        
    Returns:
        dict: Climate data for the entire boundary area
    """
    # Step 1: Download climate data for all grid points within the boundary
    logger.info("Downloading climate data for entire study area boundary")
    climate_data = download_boundary_climate_data(
        boundary_shapefile, start_date, end_date, output_dir, max_concurrent
    )
    
    # Step 2: Create PAR files for all grid points
    logger.info("Creating PAR files for climate data")
    climate_data = create_par_files_for_grid(climate_data, cligen_path)
    
    # Step 3: Add utility function to create climate files for hillslopes
    climate_data['create_climate_file'] = lambda coords, hillslope_id, temp_dir=None: create_climate_file_for_hillslope(
        climate_data, coords, hillslope_id, cligen_path, temp_dir
    )
    
    return climate_data

def download_boundary_climate_data(boundary_shapefile, start_date, end_date, output_dir, max_concurrent=10):
    """
    Download climate data for all grid points within the boundary shapefile.
    
    Args:
        boundary_shapefile (str): Path to boundary shapefile
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Directory to save output files
        max_concurrent (int): Maximum number of concurrent downloads
        
    Returns:
        dict: Climate data grid with spatial index for hillslope assignment
    """
    # Create climate directory
    climate_dir = os.path.join(output_dir, "climate")
    os.makedirs(climate_dir, exist_ok=True)
    
    # Check for cached boundary climate data
    cache_key = hashlib.md5(f"{boundary_shapefile}_{start_date}_{end_date}".encode()).hexdigest()
    cache_file = os.path.join(get_cache_dir(), f"boundary_climate_{cache_key}.json")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_meta = json.load(f)
            
            # Check if cache is recent enough (30 days)
            cache_age_days = (datetime.now() - datetime.fromisoformat(cache_meta['created'])).days
            if cache_age_days <= 30:
                logger.info(f"Using cached boundary climate data ({cache_age_days} days old)")
                
                # Load grid points from cache
                grid_points = {}
                for point_str, file_path in cache_meta['grid_points'].items():
                    lon, lat = map(float, point_str.split('_'))
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, parse_dates=['date'])
                        df.set_index('date', inplace=True)
                        grid_points[(lon, lat)] = df
                
                # Create spatial index
                idx = index.Index()
                for i, grid_point in enumerate(grid_points.keys()):
                    lon, lat = grid_point
                    idx.insert(i, (lon, lat, lon, lat))
                
                return {
                    'grid_points': grid_points,
                    'spatial_index': idx,
                    'output_dir': climate_dir,
                    'start_date': start_date,
                    'end_date': end_date,
                }
        except Exception as e:
            logger.warning(f"Failed to load climate data from cache: {e}")
    
    # Read boundary shapefile to determine grid points
    boundary = gpd.read_file(boundary_shapefile)
    if boundary.crs != 'EPSG:4326':
        boundary = boundary.to_crs('EPSG:4326')
    
    # Get bounding box
    minx, miny, maxx, maxy = boundary.total_bounds
    
    # Create prepared geometry for efficient containment tests
    prepared_boundary = prep(boundary.unary_union)
    
    # Determine optimal grid spacing based on area
    grid_spacing = 0.01  # default ~1km spacing
    area_km2 = boundary.to_crs('+proj=utm +zone=11 +datum=WGS84').area.sum() / 1e6
    
    if area_km2 > 1000:
        grid_spacing = 0.03  # ~3km spacing for large areas
    elif area_km2 > 100:
        grid_spacing = 0.02  # ~2km spacing for medium areas
    
    logger.info(f"Using grid spacing of {grid_spacing:.4f} degrees (~{grid_spacing*111:.1f}km)")
    
    # Generate grid points
    lat_points = np.arange(miny, maxy + grid_spacing, grid_spacing)
    lon_points = np.arange(minx, maxx + grid_spacing, grid_spacing)
    
    # Find grid points within boundary
    grid_points = []
    for lat in lat_points:
        for lon in lon_points:
            point = Point(lon, lat)
            if prepared_boundary.contains(point):
                grid_points.append((lon, lat))
    
    # Add boundary centroid as fallback
    centroid = boundary.unary_union.centroid
    grid_points.append((centroid.x, centroid.y))
    
    logger.info(f"Will download data for {len(grid_points)} grid points")
    
    # Set up async download
    async def download_all_grid_points():
        grid_point_data = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_download(point):
            lon, lat = point
            async with semaphore:
                try:
                    async with aiohttp.ClientSession() as session:
                        data = await download_daymet_data(lat, lon, start_date, end_date, session)
                        return point, data
                except Exception as e:
                    logger.error(f"Error downloading data for {lat}, {lon}: {e}")
                    return point, None
        
        # Create tasks for all grid points
        tasks = [bounded_download(point) for point in grid_points]
        
        # Process in batches for better progress tracking
        results = []
        batch_size = min(len(tasks), max_concurrent * 2)
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            
            logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            batch_results = await asyncio.gather(*batch)
            
            for point, data in batch_results:
                if data is not None:
                    grid_point_data[point] = data
        
        return grid_point_data
    
    # Run async download within a dedicated event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        grid_point_data = loop.run_until_complete(download_all_grid_points())
    finally:
        loop.close()
    
    logger.info(f"Successfully downloaded data for {len(grid_point_data)} grid points")
    
    if not grid_point_data:
        raise RuntimeError("Failed to download climate data for any grid points")
    
    # Create spatial index for efficient lookup
    idx = index.Index()
    for i, grid_point in enumerate(grid_point_data.keys()):
        lon, lat = grid_point
        idx.insert(i, (lon, lat, lon, lat))
    
    # Cache the results for future use
    try:
        # Create serializable representation
        serializable_grid_points = {}
        for grid_point, df in grid_point_data.items():
            lon, lat = grid_point
            point_key = f"{lon}_{lat}"
            cache_df_file = os.path.join(get_cache_dir(), f"grid_{point_key}_{start_date}_{end_date}.csv")
            df.to_csv(cache_df_file)
            serializable_grid_points[point_key] = cache_df_file
        
        # Save metadata
        cache_meta = {
            'created': datetime.now().isoformat(),
            'boundary_shapefile': boundary_shapefile,
            'start_date': start_date,
            'end_date': end_date,
            'grid_points': serializable_grid_points
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_meta, f)
        
        logger.info(f"Cached boundary climate data to: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache boundary climate data: {e}")
    
    # Return the climate data with spatial index
    return {
        'grid_points': grid_point_data,
        'spatial_index': idx,
        'output_dir': climate_dir,
        'start_date': start_date, 
        'end_date': end_date
    }

# ================================================
# SLOPE DATA PROCESSING
# ================================================

def create_slope_file(hillslope, dem_raster, output_dir, hillslope_id=None):
    """
    Create a WEPP slope file from DEM data.
    
    Args:
        hillslope (GeoSeries): Hillslope geometry
        dem_raster (str): Path to DEM raster
        output_dir (str): Directory to save output file
        hillslope_id (str, optional): Hillslope ID
        
    Returns:
        str: Path to WEPP slope file
    """
    if hillslope_id is None:
        hillslope_id = str(hillslope['id']) if 'id' in hillslope else f"hillslope_{hillslope.name}"
    
    slope_file = os.path.join(output_dir, f"slope_{hillslope_id}.slp")
    
    # Open DEM raster
    with rasterio.open(dem_raster) as src:
        # Clip DEM to hillslope boundary
        clipped_dem, clipped_transform = mask(src, [mapping(hillslope.geometry)], crop=True)
        
        # Check if we have valid data
        if clipped_dem.size == 0 or np.all(clipped_dem == src.nodata):
            raise ValueError(f"No valid DEM data for hillslope {hillslope_id}")
        
        # Extract elevation data
        dem_data = clipped_dem[0]
        dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
        
        # Calculate elevation difference and distance
        elev_max = np.nanmax(dem_data)
        elev_min = np.nanmin(dem_data)
        
        # Calculate hillslope length from geometry
        length = hillslope.geometry.length
        if length < 1:  # Ensure reasonable length
            raise ValueError(f"Hillslope {hillslope_id} has invalid length: {length}")
        
        # Calculate average slope (as a fraction, not percent)
        slope_frac = (elev_max - elev_min) / length
        
        # Ensure slope is valid
        if slope_frac < 0.001 or slope_frac > 1.0:
            logger.warning(f"Hillslope {hillslope_id} has extreme slope value: {slope_frac:.4f}, adjusting")
            slope_frac = max(0.001, min(1.0, slope_frac))
        
        # Determine aspect
        aspect = 0  # Default to north
        try:
            # Try to compute a more accurate aspect if possible
            from sklearn.linear_model import LinearRegression
            
            # Get coordinates of non-nan values
            y_indices, x_indices = np.where(~np.isnan(dem_data))
            
            if len(y_indices) > 10:  # Need enough points for regression
                coords = np.column_stack([x_indices, y_indices])
                elevs = dem_data[y_indices, x_indices]
                
                # Fit a plane to the points
                model = LinearRegression().fit(coords, elevs)
                
                # Calculate aspect from coefficients
                dx, dy = model.coef_
                aspect_rad = np.arctan2(dy, dx)
                aspect = np.degrees(aspect_rad) % 360
        except:
            # If aspect calculation fails, use hillslope's general direction
            try:
                start_point = hillslope.geometry.boundary.coords[0]
                end_point = hillslope.geometry.boundary.coords[-1]
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                aspect_rad = np.arctan2(dy, dx)
                aspect = np.degrees(aspect_rad) % 360
            except:
                pass
    
    # Write WEPP slope file according to the format in the manual
    with open(slope_file, 'w') as f:
        # Version control number
        f.write("95.7\n")
        # Number of OFEs
        f.write("1\n")
        # Aspect and width
        f.write(f"{aspect:.1f} 100\n")
        # Number of slope points and length
        f.write(f"3 {length:.2f}\n")
        # Slope points - Using 3 points to represent S-curve
        # Start with zero slope, middle with max slope, end with lower slope
        f.write(f"0.0,{slope_frac*0.8:.6f} 0.5,{slope_frac:.6f} 1.0,{slope_frac*0.5:.6f}\n")
    
    return slope_file

# ================================================
# MANAGEMENT DATA PROCESSING
# ================================================
def fetch_nlcd_data(dem_raster, output_file, temp_dir):
    """
    Fetch NLCD land cover data aligned with DEM extent.
    
    Args:
        dem_raster (str): Path to DEM raster for reference
        output_file (str): Path to save NLCD raster
        temp_dir (str): Directory for temporary files
        
    Returns:
        str: Path to NLCD raster
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info(f"Using existing NLCD data: {output_file}")
        return output_file
    
    logger.info(f"Fetching NLCD data using DEM as template")
    
    try:
        # Get DEM bounds and projection
        with rasterio.open(dem_raster) as dem:
            dem_crs = dem.crs
            dem_bounds = dem.bounds
            
            # Convert bounds to WGS84 for web request
            min_x, min_y, max_x, max_y = transform_bounds(
                dem_crs, "EPSG:4326", 
                dem_bounds.left, dem_bounds.bottom, 
                dem_bounds.right, dem_bounds.top
            )
        
        # Build NLCD request URL
        url = (
            f"https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2021_Land_Cover_L48/wcs?"
            f"service=WCS&version=2.0.1&request=getcoverage&coverageid=NLCD_2021_Land_Cover_L48"
            f"&subset=Lat({min_y},{max_y})&subset=Long({min_x},{max_x})"
            f"&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        )
        
        # Download to temporary file
        temp_file = os.path.join(temp_dir, "temp_nlcd.tif")
        
        logger.info(f"Downloading NLCD data")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"NLCD download failed with status code {response.status_code}")
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Reproject to match DEM
        logger.info(f"Reprojecting NLCD data to match DEM")
        with rasterio.open(temp_file) as src:
            # Calculate transform for reprojection
            transform, width, height = calculate_default_transform(
                src.crs, dem_crs, src.width, src.height, *src.bounds
            )
            
            # Create destination array
            dst_data = np.zeros((height, width), dtype=src.dtypes[0])
            
            # Perform reprojection
            reproject(
                source=src.read(1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dem_crs,
                resampling=Resampling.nearest  # Use nearest neighbor for categorical data
            )
            
            # Create output metadata
            out_meta = src.meta.copy()
            out_meta.update({
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'transform': transform,
                'crs': dem_crs
            })
            
            # Write the output file
            with rasterio.open(output_file, 'w', **out_meta) as dst:
                dst.write(dst_data, 1)
        
        # Remove temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger.info(f"NLCD data processing completed: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error fetching NLCD data: {e}")
        if os.path.exists(output_file) and os.path.getsize(output_file) == 0:
            os.remove(output_file)
        raise
    
    
def create_management_file(hillslope, nlcd_raster, output_dir, hillslope_id=None):
    """
    Create a WEPP management file based on NLCD land cover data.
    
    Args:
        hillslope (GeoSeries): Hillslope geometry
        nlcd_raster (str): Path to NLCD raster
        output_dir (str): Directory to save output file
        hillslope_id (str, optional): Hillslope ID
        
    Returns:
        str: Path to WEPP management file
    """
    # Assign hillslope ID if not provided
    if hillslope_id is None:
        hillslope_id = str(hillslope['id']) if 'id' in hillslope else f"hillslope_{hillslope.name}"
    
    management_file = os.path.join(output_dir, f"management_{hillslope_id}.man")
    
    # Map NLCD classes to WEPP management types - simplified to key categories
    nlcd_wepp_map = {
        # Water and ice
        11: "fallow", 12: "fallow",
        # Urban
        21: "urban_open", 22: "urban_low", 23: "urban_medium", 24: "urban_high",
        # Barren
        31: "fallow",
        # Forest
        41: "forest_evergreen", 42: "forest_deciduous", 43: "forest_mixed",
        # Shrub and grassland
        52: "shrubland", 71: "grassland",
        # Agriculture
        81: "pasture", 82: "cropland",
        # Wetlands
        90: "forest_wetland", 95: "wetland"
    }
    
    # Default management type if classification fails
    default_management = "forest_mixed"
    default_nlcd_class = 43
    
    # Get dominant land cover class
    try:
        with rasterio.open(nlcd_raster) as src:
            # Clip to hillslope boundary
            geometry = [mapping(hillslope.geometry)]
            nlcd_clip, _ = mask(src, geometry, crop=True)
            nlcd_data = nlcd_clip[0]
            
            # Count occurrences of each land cover class
            values, counts = np.unique(nlcd_data[nlcd_data != src.nodata], return_counts=True)
            if len(counts) > 0:
                dominant_lc = values[np.argmax(counts)]
                wepp_management = nlcd_wepp_map.get(dominant_lc, default_management)
                logger.info(f"Dominant NLCD class for hillslope {hillslope_id}: {dominant_lc} -> {wepp_management}")
            else:
                wepp_management = default_management
                dominant_lc = default_nlcd_class
                logger.warning(f"No valid NLCD data for hillslope {hillslope_id}, using default")
    except Exception as e:
        logger.error(f"Error processing NLCD data: {e}")
        wepp_management = default_management
        dominant_lc = default_nlcd_class
    
    # Create management parameters based on dominant land cover
    management_params = {
        "forest": {
            "plant_params": "14.0 23.0 15.0 10.0 5.0 30.0 0.0 0.15 1.0 0.5\n1700.0 0.01 0.65 0.99 12.0 0.50 2.6 2\n0.016 0.016 25.0 0.0 0.20 1.5 0.25 1.0 30 0\n0.0 6.0 0.0",
            "management_type": 2,  # perennial
            "canopy": "1.2 0.8 999 77 0 0.95",
            "initial": "999 0.05 0.95 0.034 1"
        },
        "crop": {
            "plant_params": "3.6 3.0 28.0 10.0 3.2 60.0 0.0 0.304 0.65 0.051\n0.8 0.98 0.65 0.99 0.0 1700.0 0.5 2.6 2\n0.016 0.016 25.0 0.0 0.219 1.52 0.25 0.0 30 0\n0.0 3.5 0.0",
            "management_type": 1,  # annual
            "canopy": "1.2 0.0 999 77 0 0.6",
            "initial": "999 0.02 0.60 0.034 1"
        },
        "grass": {
            "plant_params": "14.0 23.0 15.0 10.0 5.0 30.0 0.0 0.15 1.0 0.90\n1700.0 0.01 0.65 0.99 12.0 0.50 0.6 2\n0.015 0.015 20.0 0.0 0.006 0.3 0.33 0.6 14 0\n0.0 9.0 0.0",
            "management_type": 2,  # perennial
            "canopy": "1.2 0.5 999 77 0 0.80",
            "initial": "999 0.03 0.80 0.034 1"
        },
        "default": {
            "plant_params": "5.0 3.0 15.0 10.0 3.0 30.0 0.0 0.15 0.9 0.05\n0.8 0.9 0.65 0.99 0.0 1200.0 0.4 1.0 2\n0.01 0.01 20.0 0.0 0.10 0.5 0.33 0.0 14 0\n0.0 4.0 0.0",
            "management_type": 1,  # annual
            "canopy": "1.2 0.5 999 77 0 0.80",
            "initial": "999 0.03 0.80 0.034 1"
        }
    }
    
    # Select parameter set based on management type
    if "forest" in wepp_management:
        params = management_params["forest"]
        plant_name = "FOREST"
        plant_desc = "Mixed forest vegetation"
    elif "crop" in wepp_management:
        params = management_params["crop"]
        plant_name = "CROP"
        plant_desc = "Cultivated crop"
    elif "grass" in wepp_management or "pasture" in wepp_management:
        params = management_params["grass"]
        plant_name = "GRASS"
        plant_desc = "Perennial grass vegetation"
    else:
        params = management_params["default"]
        plant_name = "DEFAULT"
        plant_desc = "Default vegetation cover"
    
    # Write management file - using clearer structure
    with open(management_file, 'w') as f:
        # Header
        f.write("95.7\n")
        f.write(f"# WEPP management file for hillslope {hillslope_id}\n")
        f.write(f"# Generated from NLCD class {dominant_lc} ({wepp_management})\n")
        f.write(f"# Created on {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("1\n")  # Number of OFEs
        f.write("5\n")  # Total years in simulation
        
        # Plant section
        f.write("1\n")  # Number of plant types
        f.write(f"{plant_name}\n")  # Plant name
        f.write(f"{plant_desc}\n")
        f.write("(auto-generated from NLCD)\n")
        f.write("Based on WEPP parameters\n")
        f.write("1\n")  # Land use - crop (1)
        
        # Select appropriate units based on plant type
        f.write("tons/ac\n" if "forest" in wepp_management or "grass" in wepp_management else "bu/ac\n")
        
        # Plant parameters
        f.write(params["plant_params"] + "\n")
        
        # Operation section (same for all types)
        f.write("1\n")  # Number of operation types
        f.write("NONE\n")  # Operation name
        f.write("No tillage operations\n")
        f.write("(auto-generated)\n")
        f.write("Default parameters\n")
        f.write("1\n")  # Land use - crop (1)
        f.write("0.1 0.05 0\n")
        f.write("4\n")  # pcode - Other
        f.write("0.025 0.75 0.1 0.05 0.012 0.15 0\n")
        
        # Initial conditions section
        f.write("1\n")  # Number of initial condition scenarios
        f.write("INITIAL\n")  # Scenario name
        f.write("Initial conditions\n")
        f.write("(auto-generated)\n")
        f.write("Default parameters\n")
        f.write("1\n")  # Land use - crop (1)
        f.write(params["canopy"] + "\n")
        f.write("1\n")  # Plant scenario index
        f.write(f"{params['management_type']}\n")  # Management type - annual(1) or perennial(2)
        f.write(params["initial"] + "\n")
        f.write("1\n")  # Rill type - temporary
        f.write("0 0 0.1 0.2 0\n")
        f.write("0.4 0\n")
        
        # Surface effects section (same for all types)
        f.write("1\n")  # Number of surface effect scenarios
        f.write("SURFEFF\n")  # Scenario name
        f.write("Surface effects\n")
        f.write("(auto-generated)\n")
        f.write("Default parameters\n")
        f.write("1\n")  # Land use - crop (1)
        f.write("0\n")  # Number of operations
        
        # Contouring and drainage sections
        f.write("0\n")  # Number of contouring scenarios
        f.write("0\n")  # Number of drainage scenarios
        
        # Yearly section
        f.write("1\n")  # Number of yearly scenarios
        f.write("YEARLY\n")  # Scenario name
        f.write("Yearly management\n")
        f.write("(auto-generated)\n")
        f.write("Default parameters\n")
        f.write("1\n")  # Land use - crop (1)
        f.write("1\n")  # Plant scenario index
        f.write("1\n")  # Surface effects scenario index
        f.write("0\n")  # Contouring scenario index
        f.write("0\n")  # Drainage scenario index
        
        # Management specific yearly parameters
        f.write(f"{params['management_type']}\n")  # Management type
        
        if params['management_type'] == 2:  # Perennial vegetation
            f.write("365\n")  # Julian date for senescence
            f.write("0\n")    # Already established
            f.write("0\n")    # No growth stopping
            f.write("0.0\n")  # Row width
            f.write("3\n")    # Not harvested
        else:  # Annual crop
            f.write("300\n")  # Harvest date
            f.write("120\n")  # Planting date
            f.write("0.7\n")  # Row width
            f.write("6\n")    # Residue management
        
        # Management section
        f.write("MANAGEMENT\n")
        f.write("WEPP management\n")
        f.write("(auto-generated)\n")
        f.write("Default parameters\n")
        f.write("1\n")  # Number of OFEs
        f.write("1\n")  # Initial conditions scenario index
        f.write("5\n")  # Number of rotation repeats
        f.write("1\n")  # Years in rotation
        
        # Write 5 years of identical management
        for _ in range(5):
            f.write("1\n")  # Number of plants per year
            f.write("1\n")  # Yearly scenario index
    
    return management_file

# =======================================================================================================
# ***************** Determining Channels for each hillslopes *****************************
# ========================================================================================================
def find_corresponding_channel(hillslope, streams):
    """Find stream segment that corresponds to a hillslope."""
    # Check for intersection
    intersecting_streams = streams[streams.intersects(hillslope.geometry)]
    
    if len(intersecting_streams) == 1:
        return intersecting_streams.iloc[0]
    elif len(intersecting_streams) > 1:
        # If multiple streams, get the longest one
        intersecting_streams['length'] = intersecting_streams.geometry.length
        return intersecting_streams.sort_values(by='length', ascending=False).iloc[0]
    else:
        # If no intersection, find nearest stream
        nearest_stream_idx = streams.distance(hillslope.geometry).idxmin()
        return streams.loc[nearest_stream_idx]

# ================================================
# WEPP MODEL EXECUTION
# ================================================

def prepare_wepp_input(hillslope_id, slope_file, soil_file, management_file, climate_file, 
                      output_dir, simulation_years=10):
    """
    Prepare WEPP input run file.
    
    Args:
        hillslope_id (str): Hillslope ID
        slope_file (str): Path to slope file
        soil_file (str): Path to soil file
        management_file (str): Path to management file
        climate_file (str): Path to climate file
        output_dir (str): Directory for output files
        simulation_years (int): Number of years to simulate
        
    Returns:
        tuple: (input_file_path, output_file_path)
    """
    input_file = os.path.join(output_dir, f"wepp_input_{hillslope_id}.txt")
    output_file = os.path.join(output_dir, f"wepp_output_{hillslope_id}.txt")
    
    # Create WEPP input file according to the format in the manual
    with open(input_file, 'w') as f:
        f.write("95.7\n")  # WEPP version
        f.write("1\n")  # Exit on errors? (1=yes)
        f.write("1\n")  # Simulation type (1=continuous)
        f.write("1\n")  # Model mode (1=hillslope)
        f.write("Yes\n")  # Create pass file?
        f.write(f"{os.path.join(output_dir, f'pass_{hillslope_id}.txt')}\n")  # Pass file path
        f.write("1\n")  # Output type (1=annual, abbreviated)
        f.write("No\n")  # Warmup output?
        f.write(f"{output_file}\n")  # Summary output
        f.write("No\n")  # Water output?
        f.write("No\n")  # Crop output?
        f.write("No\n")  # Soil output?
        f.write("Yes\n")  # Plotting output?
        f.write(f"{os.path.join(output_dir, f'plot_{hillslope_id}.txt')}\n")  # Plotting output path
        f.write("No\n")  # Graphics output?
        f.write("No\n")  # Event/OFE output?
        f.write("No\n")  # Event/OFE output?
        f.write("No\n")  # Event/OFE output?
        f.write("No\n")  # Winter output?
        f.write("No\n")  # Yield output?
        f.write(f"{management_file}\n")  # Management file
        f.write(f"{slope_file}\n")  # Slope file
        f.write(f"{climate_file}\n")  # Climate file
        f.write(f"{soil_file}\n")  # Soil file
        f.write("0\n")  # No irrigation
        f.write(f"{simulation_years}\n")  # Number of years to simulate
        f.write("0\n")  # All events (0 = don't bypass small events)
    
    return input_file, output_file

def run_wepp_model(input_file, wepp_path):
    """
    Run WEPP model with specified input file.
    
    Args:
        input_file (str): Path to WEPP input file
        wepp_path (str): Path to WEPP executable
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Running WEPP with input file: {input_file}")
    
    try:
        # Ensure WEPP executable has correct permissions
        if not os.access(wepp_path, os.X_OK):
            os.chmod(wepp_path, 0o755)
        
        # Run WEPP
        with open(input_file, 'r') as f_in:
            process = subprocess.run(
                [wepp_path],
                stdin=f_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
        
        # Check for errors
        if process.returncode != 0:
            logger.error(f"WEPP execution failed with return code {process.returncode}")
            logger.error(f"WEPP stderr: {process.stderr}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error running WEPP: {e}")
        return False

def parse_wepp_output(output_file, hillslope_id):
    """
    Parse WEPP output file to extract results.
    
    Args:
        output_file (str): Path to WEPP output file
        hillslope_id (str): Hillslope ID
        
    Returns:
        dict: Parsed results
    """
    results = {
        'hillslope_id': hillslope_id,
        'precipitation_mm': 0,
        'runoff_mm': 0,
        'soil_loss_kg_m2': 0,
        'sediment_yield_kg': 0,  # Changed from sediment_yield_t to sediment_yield_kg
        'annual_values': []
    }
    
    try:
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.warning(f"Empty or missing output file: {output_file}")
            return results
        
        with open(output_file, 'r') as f:
            content = f.read()
            
            # Extract average annual results
            in_annual_summary = False
            annual_section = ""
            
            for line in content.split('\n'):
                if 'AVERAGE ANNUAL TOTALS' in line:
                    in_annual_summary = True
                    continue
                
                if in_annual_summary:
                    annual_section += line + '\n'
                    
                    # Extract key values
                    if 'PRECIPITATION' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            results['precipitation_mm'] = float(val)
                    
                    elif 'RUNOFF' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            results['runoff_mm'] = float(val)
                    
                    elif 'SOIL LOSS' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            results['soil_loss_kg_m2'] = float(val)
                    
                    elif 'SEDIMENT YIELD' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            # Convert tons to kg (1 ton = 1000 kg)
                            results['sediment_yield_kg'] = float(val) * 1000
            
            # Extract annual data
            annual_data = []
            in_yearly_data = False
            yearly_section = ""
            
            for line in content.split('\n'):
                if 'ANNUAL SUMMARY' in line:
                    in_yearly_data = True
                    yearly_section = line + '\n'
                    continue
                
                if in_yearly_data:
                    yearly_section += line + '\n'
                    
                    # Extract year data from table rows
                    if line.strip() and line[0].isdigit():
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                year_data = {
                                    'year': int(parts[0]),
                                    'precipitation_mm': float(parts[1]),
                                    'runoff_mm': float(parts[2]),
                                    'soil_loss_kg_m2': float(parts[3])
                                }
                                annual_data.append(year_data)
                            except (ValueError, IndexError):
                                pass
            
            results['annual_values'] = annual_data
        
        return results
    
    except Exception as e:
        logger.error(f"Error parsing WEPP output: {e}")
        return results

def generate_summary_report(results, output_dir):
    """
    Generate a summary report of WEPP results.
    
    Args:
        results (dict): WEPP results dictionary
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the summary report
    """
    summary_file = os.path.join(output_dir, "wepp_summary.csv")
    
    # Create summary dataframes more efficiently
    hillslope_data = []
    channel_data = []
    
    # Process hillslope results
    for hillslope_id, data in results['hillslopes'].items():
        if 'error' in data:
            continue
            
        hillslope_data.append({
            'hillslope_id': hillslope_id,
            'area_ha': data.get('area_ha', 0),
            'precipitation_mm': data.get('precipitation_mm', 0),
            'runoff_mm': data.get('runoff_mm', 0),
            'soil_loss_kg_m2': data.get('soil_loss_kg_m2', 0),
            'sediment_yield_kg': data.get('sediment_yield_kg', 0)
        })
    
    # Process channel results
    for channel_id, data in results['channels'].items():
        if 'error' in data:
            continue
            
        channel_data.append({
            'channel_id': channel_id,
            'channel_runoff_mm': data.get('channel_runoff', 0),
            'channel_sediment_kg': data.get('channel_sediment', 0),
            'peak_flow_m3s': data.get('peak_flow', 0)
        })
    
    # Create and save dataframes - use try/except for better error handling
    try:
        if hillslope_data:
            hillslope_df = pd.DataFrame(hillslope_data)
            hillslope_output = os.path.join(output_dir, "hillslope_summary.csv")
            hillslope_df.to_csv(hillslope_output, index=False)
            logger.info(f"Hillslope summary saved to: {hillslope_output}")
        
        if channel_data:
            channel_df = pd.DataFrame(channel_data)
            channel_output = os.path.join(output_dir, "channel_summary.csv")
            channel_df.to_csv(channel_output, index=False)
            logger.info(f"Channel summary saved to: {channel_output}")
        
        # Create watershed total summary
        if hillslope_data:
            # Use numpy for more efficient calculations
            import numpy as np
            
            areas = np.array([row['area_ha'] for row in hillslope_data])
            total_area = np.sum(areas)
            
            if total_area > 0:
                precip_values = np.array([row['precipitation_mm'] for row in hillslope_data])
                runoff_values = np.array([row['runoff_mm'] for row in hillslope_data])
                
                # Weighted averages
                avg_precip = np.sum(precip_values * areas) / total_area
                avg_runoff = np.sum(runoff_values * areas) / total_area
                
                # Total calculations
                soil_loss_values = np.array([row['soil_loss_kg_m2'] for row in hillslope_data])
                total_soil_loss = np.sum(soil_loss_values * areas * 10000)  # kg/m² * ha * 10000 m²/ha
                
                sediment_values = np.array([row['sediment_yield_kg'] for row in hillslope_data])
                total_sediment = np.sum(sediment_values)
                
                # Add channel sediment contribution if available
                if channel_data:
                    channel_sediment = np.sum([row['channel_sediment_kg'] for row in channel_data])
                    total_channel_sediment = channel_sediment
                else:
                    total_channel_sediment = 0
                
                avg_soil_loss = total_soil_loss / (total_area * 10000) if total_area > 0 else 0
                sediment_delivery_ratio = total_sediment / total_soil_loss if total_soil_loss > 0 else 0
                
                # Calculate watershed sediment yield
                watershed_sediment_yield = total_sediment + total_channel_sediment
            else:
                avg_precip = avg_runoff = total_soil_loss = total_sediment = avg_soil_loss = sediment_delivery_ratio = 0
                total_channel_sediment = watershed_sediment_yield = 0
            
            watershed_summary = {
                'total_area_ha': total_area,
                'avg_precipitation_mm': avg_precip,
                'avg_runoff_mm': avg_runoff,
                'total_soil_loss_kg': total_soil_loss,
                'total_hillslope_sediment_kg': total_sediment,
                'total_channel_sediment_kg': total_channel_sediment,
                'watershed_sediment_yield_kg': watershed_sediment_yield,
                'avg_soil_loss_kg_m2': avg_soil_loss,
                'sediment_delivery_ratio': sediment_delivery_ratio
            }
            
            # Save watershed summary
            watershed_output = os.path.join(output_dir, "watershed_summary.csv")
            pd.DataFrame([watershed_summary]).to_csv(watershed_output, index=False)
            logger.info(f"Watershed summary saved to: {watershed_output}")
            
            # Generate summary report with formatted text
            with open(summary_file, 'w') as f:
                f.write("WEPP Watershed Analysis Summary\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Number of hillslopes: {len(hillslope_data)}\n")
                f.write(f"Total watershed area: {total_area:.2f} ha\n")
                f.write(f"Average precipitation: {avg_precip:.2f} mm\n")
                f.write(f"Average runoff: {avg_runoff:.2f} mm\n")
                f.write(f"Total soil loss: {total_soil_loss:.2f} kg\n")
                f.write(f"Total hillslope sediment yield: {total_sediment:.2f} kg\n")
                
                if channel_data:
                    f.write(f"Total channel sediment contribution: {total_channel_sediment:.2f} kg\n")
                    f.write(f"Watershed sediment yield: {watershed_sediment_yield:.2f} kg\n")
                
                f.write(f"Average soil loss rate: {avg_soil_loss:.4f} kg/m²\n")
                f.write(f"Sediment delivery ratio: {sediment_delivery_ratio:.4f}\n\n")
                
                if channel_data:
                    channel_runoff_values = np.array([row['channel_runoff_mm'] for row in channel_data])
                    channel_sediment_values = np.array([row['channel_sediment_kg'] for row in channel_data])
                    peak_flow_values = np.array([row['peak_flow_m3s'] for row in channel_data])
                    
                    avg_channel_runoff = np.mean(channel_runoff_values)
                    total_channel_sediment = np.sum(channel_sediment_values)
                    max_peak_flow = np.max(peak_flow_values) if len(peak_flow_values) > 0 else 0
                    
                    f.write(f"Number of channels: {len(channel_data)}\n")
                    f.write(f"Average channel runoff: {avg_channel_runoff:.2f} mm\n")
                    f.write(f"Total channel sediment: {total_channel_sediment:.2f} kg\n")
                    f.write(f"Maximum peak flow: {max_peak_flow:.2f} m³/s\n")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        logger.error(traceback.format_exc())
        with open(summary_file, 'w') as f:
            f.write(f"Error generating summary: {str(e)}\n")
    
    return summary_file

def run_wepp_processing(
    hillslope_shapefile, 
    wepp_stream_shapefile, 
    soil_data_gdf,
    nlcd_raster_path, 
    dem_raster, 
    output_dir, 
    wepp_path,
    start_date, 
    end_date, 
    simulation_years, 
    max_workers,
    temp_dir_base, 
    cligen_path, 
    climate_data  # This is already processed for the entire boundary
):
    """
    Process WEPP runs for all hillslopes in parallel.
    Each hillslope uses the nearest climate grid point from the boundary-wide climate data.
    
    Args:
        hillslope_shapefile (str): Path to hillslope shapefile
        wepp_stream_shapefile (str): Path to stream shapefile
        soil_data_gdf (GeoDataFrame): Processed soil data GeoDataFrame
        nlcd_raster_path (str): Path to NLCD raster
        dem_raster (str): Path to DEM raster
        output_dir (str): Directory to save outputs
        wepp_path (str): Path to WEPP executable
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        simulation_years (int): Number of years to simulate
        max_workers (int): Maximum number of parallel workers
        temp_dir_base (str): Base directory for temporary files
        cligen_path (str): Path to CLIGEN executable
        climate_data (dict): Processed climate data object for entire boundary
        
    Returns:
        dict: Results dictionary with hillslope and channel results
    """
    # Load hillslopes data
    hillslopes_gdf = gpd.read_file(hillslope_shapefile)
    
    # Create result containers
    results = {'hillslopes': {}, 'channels': {}}

    def process_single_hillslope(hillslope_row_tuple):
        """Process a single hillslope and return its results."""
        index, hillslope = hillslope_row_tuple
        hillslope_id = str(hillslope['id']) if 'id' in hillslope else f"hillslope_{index}"
        
        # Get thread-local temporary directory
        thread_temp_dir = get_thread_temp_dir(base_temp_dir=temp_dir_base)

        try:
            logger.info(f"Processing hillslope: {hillslope_id}")
            
            # 1. Create slope file
            slope_file = create_slope_file(hillslope, dem_raster, thread_temp_dir, hillslope_id)
            
            # 2. Create soil file (using processed soil data)
            soil_file = create_soil_file(hillslope, soil_data_gdf, thread_temp_dir, hillslope_id)
            
            # 3. Create management file
            management_file = create_management_file(hillslope, nlcd_raster_path, thread_temp_dir, hillslope_id)
            
            # 4. Create climate file using climate data from the boundary
            centroid = hillslope.geometry.centroid
            
            # Ensure coordinates are in WGS84 (EPSG:4326) for climate functions
            if hillslopes_gdf.crs != "EPSG:4326":
                # Convert coordinates to WGS84
                hillslope_point = gpd.GeoDataFrame(
                    geometry=[centroid], 
                    crs=hillslopes_gdf.crs
                ).to_crs("EPSG:4326")
                coords = (hillslope_point.geometry.x[0], hillslope_point.geometry.y[0])
            else:
                coords = (centroid.x, centroid.y)
            
            # Use the climate data object's climate file creation function
            # This will find the nearest climate grid point from the boundary-wide data
            climate_output_dir = ensure_dir(os.path.join(output_dir, "climate_files"))
            cli_file = climate_data['create_climate_file'](coords, hillslope_id, thread_temp_dir)

            # 5. Prepare WEPP input files
            wepp_model_output_dir = ensure_dir(os.path.join(output_dir, "wepp_model_outputs"))
            input_run_file, output_summary_file = prepare_wepp_input(
                hillslope_id, 
                slope_file, 
                soil_file, 
                management_file, 
                cli_file,
                wepp_model_output_dir,
                simulation_years
            )

            # 6. Run WEPP model
            success = run_wepp_model(input_run_file, wepp_path)
            if not success:
                return hillslope_id, {'error': 'WEPP run failed'}

            # 7. Parse output
            parsed_results = parse_wepp_output(output_summary_file, hillslope_id)
            
            # Add area information
            parsed_results['area_ha'] = hillslope.geometry.area / 10000 if hillslopes_gdf.crs.is_projected else None

            return hillslope_id, parsed_results
            
        except Exception as e:
            logger.error(f"Error processing hillslope {hillslope_id}: {e}")
            logger.error(traceback.format_exc())
            return hillslope_id, {'error': str(e)}

    # Determine optimal number of worker threads
    num_workers = determine_optimal_workers(override=max_workers)
    logger.info(f"Using {num_workers} workers for WEPP processing.")

    # Process hillslopes in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_hillslope = {
            executor.submit(process_single_hillslope, (index, row)): (index, row)
            for index, row in hillslopes_gdf.iterrows()
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_hillslope), 
                          total=len(hillslopes_gdf), desc="Processing Hillslopes"):
            hillslope_id, result_data = future.result()
            results['hillslopes'][hillslope_id] = result_data
    
    # Process channels by linking them to hillslopes
    if os.path.exists(wepp_stream_shapefile):
        streams_gdf = gpd.read_file(wepp_stream_shapefile)
        
        for hillslope_id, hillslope_data in results['hillslopes'].items():
            if 'error' in hillslope_data:
                continue
                
            # Find hillslope in the geodataframe
            hillslope_idx = None
            for idx, row in hillslopes_gdf.iterrows():
                row_id = str(row['id']) if 'id' in row else f"hillslope_{idx}"
                if row_id == hillslope_id:
                    hillslope_idx = idx
                    break
            
            if hillslope_idx is not None:
                hillslope = hillslopes_gdf.loc[hillslope_idx]
                try:
                    # Find corresponding channel
                    channel = find_corresponding_channel(hillslope, streams_gdf)
                    channel_id = str(channel['id']) if 'id' in channel else f"channel_{channel.name}"
                    
                    # Add runoff data from hillslope to corresponding channel
                    if channel_id not in results['channels']:
                        results['channels'][channel_id] = {
                            'channel_runoff': hillslope_data.get('runoff_mm', 0) * 0.9,  # Assume some loss
                            'channel_sediment': hillslope_data.get('sediment_yield_kg', 0) * 0.8,  # Assume some deposition
                            'peak_flow': hillslope_data.get('runoff_mm', 0) * 0.05  # Approximate peak flow
                        }
                    else:
                        # Add contribution from this hillslope
                        results['channels'][channel_id]['channel_runoff'] += hillslope_data.get('runoff_mm', 0) * 0.9
                        results['channels'][channel_id]['channel_sediment'] += hillslope_data.get('sediment_yield_kg', 0) * 0.8
                        # Use max for peak flow
                        results['channels'][channel_id]['peak_flow'] = max(
                            results['channels'][channel_id]['peak_flow'],
                            hillslope_data.get('runoff_mm', 0) * 0.05
                        )
                except Exception as e:
                    logger.error(f"Error processing channel for hillslope {hillslope_id}: {e}")

    return results


def process_watershed_for_wepp(
    hillslope_shapefile, 
    wepp_stream_shapefile, 
    soil_shapefile, 
    nlcd_raster, 
    dem_raster,
    watershed_shapefile,
    output_dir, 
    wepp_path,
    temp_dir_path,
    cligen_path,
    boundary_shapefile,  # Overall boundary, different from watershed_shapefile
    start_date=None, 
    end_date=None, 
    simulation_years=10,
    max_workers=None
):
    """
    Process watershed data and run WEPP model to generate erosion outputs.
    
    Hierarchy:
    - Boundary shapefile: Overall study area boundary
    - Watershed shapefile: Multiple watersheds within the boundary
    - Hillslope shapefile: Multiple hillslopes within each watershed
    
    Climate data is processed once for the entire boundary and then assigned
    to each hillslope based on proximity.
    
    Args:
        hillslope_shapefile (str): Path to hillslope boundary shapefile (smallest units)
        wepp_stream_shapefile (str): Path to stream vector file
        soil_shapefile (str): Path to GSSURGO soil data shapefile
        nlcd_raster (str): Path to NLCD land cover raster
        dem_raster (str): Path to Digital Elevation Model
        watershed_shapefile (str): Path to watershed boundary shapefile (contains multiple hillslopes)
        output_dir (str): Directory to save WEPP outputs
        wepp_path (str): Path to WEPP executable
        temp_dir_path (str): Path for temporary directory
        cligen_path (str): Path to CLIGEN executable
        boundary_shapefile (str): Path to overall boundary shapefile (contains multiple watersheds)
        start_date (str, optional): Start date for climate data in YYYY-MM-DD format
        end_date (str, optional): End date for climate data in YYYY-MM-DD format
        simulation_years (int, optional): Number of years to simulate in WEPP
        max_workers (int, optional): Maximum number of worker threads
    
    Returns:
        tuple: (watershed_gdf, hillslope_gdf, channel_gdf) - GeoDataFrames with WEPP results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365*simulation_years)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch and save the NLCD raster for the region
    # Use boundary shapefile to determine the extent
    nlcd_output = os.path.join(output_dir, "nlcd_data.tif")
    if not os.path.exists(nlcd_output):
        logger.info("Fetching NLCD data with DEM template...")
        try:
            nlcd_output = fetch_nlcd_data(dem_raster, nlcd_output, temp_dir_path)
            logger.info(f"NLCD data saved to: {nlcd_output}")
        except Exception as e:
            logger.error(f"Error fetching NLCD data: {e}")
            logger.error(traceback.format_exc())
            nlcd_output = nlcd_raster  # Fall back to original data
    
    # Process soil data - use boundary shapefile for the overall extent
    logger.info("Processing soil data from GSSURGO...")
    try:
        soil_output_dir = os.path.join(output_dir, "soil_data")
        os.makedirs(soil_output_dir, exist_ok=True)
        soil_data_gdf = process_gssurgo_data(soil_shapefile, boundary_shapefile, soil_output_dir)
        logger.info("Successfully processed soil data")
    except Exception as e:
        logger.error(f"Error processing soil data: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
    
    # Process climate data for the entire boundary area (not per watershed)
    logger.info("Processing climate data for the entire study area boundary...")
    try:
        climate_data = process_watershed_climate(
            boundary_shapefile,  # Use boundary for overall climate grid 
            start_date, 
            end_date, 
            output_dir, 
            cligen_path, 
            max_concurrent=determine_optimal_workers(override=max_workers)
        )
        logger.info("Successfully processed climate data")
    except Exception as e:
        logger.error(f"Error processing climate data: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
    
    # Execute the WEPP processing for all hillslopes, using boundary-wide climate data
    try:
        results = run_wepp_processing(
            hillslope_shapefile, 
            wepp_stream_shapefile, 
            soil_data_gdf,
            nlcd_output, 
            dem_raster, 
            output_dir, 
            wepp_path,
            start_date, 
            end_date, 
            simulation_years, 
            max_workers,
            temp_dir_path,
            cligen_path,
            climate_data  # Boundary-wide climate data
        )
        
        if 'error' in results:
            logger.error(f"Error in WEPP processing: {results['error']}")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Error running WEPP processing: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

    # Now create the GeoDataFrames with the results
    try:
        # 1. Read the watershed boundaries
        watersheds_gdf = gpd.read_file(watershed_shapefile)
        
        # Add columns to store results for each watershed
        watersheds_gdf['total_sediment_kg'] = 0
        watersheds_gdf['total_runoff_mm'] = 0
        watersheds_gdf['peak_flow_m3s'] = 0
        
        # 2. Read hillslopes and streams for processing
        streams_gdf = gpd.read_file(wepp_stream_shapefile)
        hillslopes_gdf = gpd.read_file(hillslope_shapefile)
        
        # 3. Associate hillslopes with their watersheds first
        # This is the key step - hillslopes are within watersheds
        hillslopes_gdf = gpd.sjoin(hillslopes_gdf, watersheds_gdf[['geometry', 'id']], 
                                  how="left", predicate="within")
        hillslopes_gdf = hillslopes_gdf.rename(columns={'id_right': 'watershed_id'})
        
        # 4. Apply WEPP results to hillslopes
        hillslopes_gdf['runoff_mm'] = 0
        hillslopes_gdf['soil_loss_kg_m2'] = 0
        hillslopes_gdf['sediment_yield_kg'] = 0
        hillslopes_gdf['precipitation_mm'] = 0
        
        for hillslope_idx, hillslope in hillslopes_gdf.iterrows():
            hillslope_id = str(hillslope['id_left']) if 'id_left' in hillslope else (
                str(hillslope['id']) if 'id' in hillslope else f"hillslope_{hillslope_idx}")
            
            if hillslope_id in results['hillslopes']:
                hillslope_result = results['hillslopes'][hillslope_id]
                if 'error' not in hillslope_result:
                    hillslopes_gdf.at[hillslope_idx, 'runoff_mm'] = hillslope_result.get('runoff_mm', 0)
                    hillslopes_gdf.at[hillslope_idx, 'soil_loss_kg_m2'] = hillslope_result.get('soil_loss_kg_m2', 0)
                    hillslopes_gdf.at[hillslope_idx, 'sediment_yield_kg'] = hillslope_result.get('sediment_yield_kg', 0)
                    hillslopes_gdf.at[hillslope_idx, 'precipitation_mm'] = hillslope_result.get('precipitation_mm', 0)
        
        # 5. Aggregate hillslope results to their parent watersheds
        # Group by watershed_id and sum/average the results
        watershed_aggregates = hillslopes_gdf.groupby('watershed_id').agg({
            'runoff_mm': 'mean',
            'precipitation_mm': 'mean',
            'soil_loss_kg_m2': 'mean',
            'sediment_yield_kg': 'sum',
            'geometry': 'count'  # Count hillslopes per watershed
        }).reset_index()
        watershed_aggregates = watershed_aggregates.rename(columns={'geometry': 'hillslope_count'})
        
        # Update the watershed GeoDataFrame with aggregated results
        for _, watershed_agg in watershed_aggregates.iterrows():
            watershed_id = watershed_agg['watershed_id']
            if pd.isna(watershed_id):
                continue
                
            watershed_idx = watersheds_gdf[watersheds_gdf['id'] == watershed_id].index
            if len(watershed_idx) > 0:
                watersheds_gdf.at[watershed_idx[0], 'total_sediment_kg'] = watershed_agg['sediment_yield_kg']
                watersheds_gdf.at[watershed_idx[0], 'total_runoff_mm'] = watershed_agg['runoff_mm']
                # Other fields can be added here
        
        # 6. Associate streams with watersheds
        channels_gdf = gpd.sjoin(streams_gdf, watersheds_gdf[['geometry', 'id']], 
                               how="left", predicate="intersects")
        channels_gdf = channels_gdf.rename(columns={'id_right': 'watershed_id'})
        
        # 7. Apply channel results
        channels_gdf['channel_runoff_mm'] = 0
        channels_gdf['channel_sediment_kg'] = 0
        channels_gdf['peak_flow_m3s'] = 0
        
        for channel_idx, channel in channels_gdf.iterrows():
            channel_id = str(channel['id_left']) if 'id_left' in channel else (
                str(channel['id']) if 'id' in channel else f"channel_{channel_idx}")
            
            if channel_id in results['channels']:
                channel_result = results['channels'][channel_id]
                if 'error' not in channel_result:
                    channels_gdf.at[channel_idx, 'channel_runoff_mm'] = channel_result.get('channel_runoff', 0)
                    channels_gdf.at[channel_idx, 'channel_sediment_kg'] = channel_result.get('channel_sediment', 0)
                    channels_gdf.at[channel_idx, 'peak_flow_m3s'] = channel_result.get('peak_flow', 0)
        
        # 8. Save the results
        watersheds_output = os.path.join(output_dir, "watersheds_with_results.gpkg")
        hillslopes_output = os.path.join(output_dir, "hillslopes_with_results.gpkg")
        channels_output = os.path.join(output_dir, "channels_with_results.gpkg")
        
        watersheds_gdf.to_file(watersheds_output, driver="GPKG")
        hillslopes_gdf.to_file(hillslopes_output, driver="GPKG")
        channels_gdf.to_file(channels_output, driver="GPKG")
        
        # Generate summary report
        summary_file = generate_summary_report(results, output_dir)
        logger.info(f"Summary report generated: {summary_file}")
        
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Watershed results: {watersheds_output}")
        logger.info(f"Hillslope results: {hillslopes_output}")
        logger.info(f"Channel results: {channels_output}")
        
        return watersheds_gdf, hillslopes_gdf, channels_gdf
        
    except Exception as e:
        logger.error(f"Error creating output GeoDataFrames: {e}")
        logger.error(traceback.format_exc())
        return None, None, None