import urllib
# from IPython.display import display
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio import plot
from rasterio.features import rasterize, shapes
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import mapping, box, shape
import requests
import os
from osgeo import gdal, ogr, osr
import zipfile
import whitebox
import shutil
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.errors import GeometryTypeError
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform
import pyproj
import folium
from folium.plugins import MeasureControl, Draw, Fullscreen
import random
# Initializing whitebox tools
wbt = whitebox.WhiteboxTools()
from app import TaskCancelledError



"""
=========================================================================================================================================
                                            RATIONAL METHOD : Definition

    The Rational Method, a simplified approach to estimating peak runoff rates, is based on several key assumptions:

        - Uniform Rainfall Intensity: The rainfall intensity is assumed to be uniform over the entire watershed for a duration equal to or greater than the time of concentration.
        
        - Time of Concentration: The time of concentration (Tc) is the time required for runoff from the most remote point in the watershed to reach the outlet. It is assumed to be constant and independent of rainfall intensity.
        
        - Negligible Storage Effects: The effects of storage in the watershed, such as infiltration and depression storage, are assumed to be negligible.
        
        - Invariant Runoff Coefficient: The runoff coefficient (C) is assumed to be constant and independent of rainfall intensity and antecedent moisture conditions.
        
        - Small Watershed: The Rational Method is best suited for small watersheds, typically less than 200 acres (80 hectares). For larger watersheds, more complex methods may be necessary.
        
        - Overland Flow Dominance: The method assumes that overland flow is the dominant flow path, neglecting subsurface flow and channel flow.
        
        - Constant Rainfall Duration: The rainfall duration is assumed to be equal to the time of concentration.

                    Q = CIA
                        where:
                        Q = Peak flow, ft3/s.
                        C = Runoff coefficient (dimensionless).                                              
                        I = Rainfall intensity, in/hr.
                        A = Drainage area, acres.

Three conditions based on data availibility

    User has neither flow nor precipitation data
    User has annual maxima of hourly or sub-hourly precipitation intensity data only
    User has both inst. flow and hourly or sub-hourly precipitation intensity data
    ========================================================================================================================================
"""
# ============================================================================================================================================
# Function to download data using url
# ============================================================================================================================================
def download_data(url, file_path,verbose=True):
  """Downloads a data file from a given URL to a specified file path.

  Args:
      url (str): The URL of the data file.
      file_path (str): The desired file path for the downloaded file, including filename.

  Raises:
      requests.exceptions.RequestException: If there's an error during the download.
  """
  # Extract filename from path (optional, for logging/messages)
  filename = os.path.basename(file_path)
  try:
      response = requests.get(url)
      response.raise_for_status()

      

      with open(file_path, 'wb') as f:
          f.write(response.content)
      if (verbose==True):
        print(f"Successfully downloaded {filename} to {file_path}")

  except requests.exceptions.RequestException as e:
      print(f"Error downloading {filename}: {e}")
      raise

 
# ================================================================================================================================================
# Extract the states abbreviations that the region - ploygon intersects with
# ===========================================================================================================================================
def get_us_states_crossed(polygon_path, usa_states_shapefile_path, state_abbr_column="stusps"
    ):
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
            states_gdf = gpd.read_file(f"zip://{usa_states_shapefile_path}").to_crs(
                "EPSG:4326"
            )

            # Create a unified boundary geometry (in case there are multiple features)
            boundary_union = boundary_gdf.geometry.unary_union

            # Find all states that intersect with the boundary polygon
            intersecting_states = states_gdf[states_gdf.intersects(boundary_union)]

            if intersecting_states.empty:
                return []

            if state_abbr_column not in intersecting_states.columns:
                print(
                    f"Column '{state_abbr_column}' not found in the states shapefile. Available columns: {intersecting_states.columns.tolist()}"
                )
                return []

            # Extract the unique state abbreviations from the intersecting states
            state_abbr_list = intersecting_states[state_abbr_column].unique().tolist()
            return state_abbr_list

        except Exception as e:
            print(f"An error occurred: {e}")
            return []
       
# =======================================================================================================================
# clip and merge the gssurgo dominant hydrologic group variable to use it in determining the CN values for each watershed
# =========================================================================================================================
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.features import shapes
from shapely.geometry import shape
import os
import pandas as pd
from pathlib import Path
import numpy as np
import subprocess
import tempfile
import shutil

def clip_raster_to_polygon(raster_path: str, polygon_path: str, output_path: str) -> None:
    # Reproject polygon to EPSG:5070
    poly = gpd.read_file(polygon_path).to_crs('EPSG:5070')
    
    # Fix invalid geometries (self-intersections, etc.)
    poly['geometry'] = poly['geometry'].buffer(0)
    
    # Create temp directory and file path for shapefile
    temp_dir = tempfile.mkdtemp()
    temp_poly = os.path.join(temp_dir, 'temp_polygon.shp')
    poly.to_file(temp_poly)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Clip raster using reprojected polygon - capture error output
    result = subprocess.run(['gdalwarp', '-cutline', temp_poly, '-crop_to_cutline',
                           '-of', 'GTiff', '-co', 'COMPRESS=LZW', raster_path, output_path], 
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"gdalwarp stdout: {result.stdout}")
        print(f"gdalwarp stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, result.args)
    
    # Clean up temp directory and all files
    shutil.rmtree(temp_dir)

def gssurgo_raster_to_clipped_polygon(polygon_path, base_raster_path, output_path, 
                             usa_states_shapefile_path, state_abbr_column="stusps"):
    """
    Converts raster data to polygon shapefile after clipping to boundary polygon.
    Handles multiple states by merging their raster data.
    
    Args:
        polygon_path (str): Path to the shapefile containing the boundary polygon
        base_raster_path (str): Base directory path like "/path/to/Soil_GSSURGO"
        output_path (str): Path where the output shapefile will be saved
        usa_states_shapefile_path (str): Path to US states shapefile
        state_abbr_column (str): Column name for state abbreviations
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the boundary polygon first and preserve its original CRS
        boundary_gdf = gpd.read_file(polygon_path)
        original_crs = boundary_gdf.crs
        boundary_union = boundary_gdf.geometry.unary_union
        
        # Get the list of states that intersect with the boundary polygon
        state_abbr_list = get_us_states_crossed(polygon_path, usa_states_shapefile_path, state_abbr_column)
        
        if not state_abbr_list:
            print("No states found that intersect with the boundary polygon.")
            return False
        
        print(f"Found {len(state_abbr_list)} states: {state_abbr_list}")
        
        all_polygons = []
        
        for state_abbr in state_abbr_list:
            # Construct the raster path for this state: base_path/state/hydgrpdcd.tif
            raster_path = os.path.join(base_raster_path, state_abbr, "hydgrpdcd.tif")
            
            if not os.path.exists(raster_path):
                print(f"Raster file not found for state {state_abbr}: {raster_path}")
                continue
            
            print(f"Processing raster for state {state_abbr}: {raster_path}")
            
            # Create temporary clipped raster path
            temp_clipped_raster = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
            
            try:
                # Clip the raster using the efficient function (no reading before clipping)
                clip_raster_to_polygon(raster_path, polygon_path, temp_clipped_raster)
                
                # Now read only the clipped raster (much smaller)
                with rasterio.open(temp_clipped_raster) as src:
                    clipped_data = src.read(1)
                    clipped_transform = src.transform
                    
                    # Convert clipped raster to polygons
                    mask = clipped_data != src.nodata if src.nodata is not None else None
                    
                    # Generate shapes from the clipped raster
                    polygon_shapes = list(shapes(clipped_data, mask=mask, transform=clipped_transform))
                    
                    # Convert to GeoDataFrame
                    geometries = []
                    values = []
                    
                    for geom, value in polygon_shapes:
                        # Skip nodata values first
                        if value == src.nodata:
                            continue
                        
                        # Convert numeric values to hydrologic soil group letters
                        if value == 1:
                            hydgrp_value = 'A'
                        elif value == 2:
                            hydgrp_value = 'B'
                        elif value == 3:
                            hydgrp_value = 'C'
                        elif value == 4:
                            hydgrp_value = 'D'
                        elif value == -1:
                            hydgrp_value = None  # Skip missing values
                            continue
                        else:
                            hydgrp_value = str(value)  # Keep other values as strings
                        
                        geometries.append(shape(geom))
                        values.append(hydgrp_value)
                    
                    if geometries:
                        # Create GeoDataFrame in raster CRS
                        state_gdf = gpd.GeoDataFrame({
                            'hydgrpdcd': values,
                            'state': state_abbr
                        }, geometry=geometries, crs=src.crs)
                        
                        # Reproject to target CRS
                        state_gdf = state_gdf.to_crs(original_crs)
                        all_polygons.append(state_gdf)
                        print(f"Extracted {len(state_gdf)} polygons from {state_abbr}")
            
            finally:
                # Clean up temporary clipped raster
                if os.path.exists(temp_clipped_raster):
                    os.remove(temp_clipped_raster)
        
        if not all_polygons:
            print("No valid polygons extracted from any state rasters.")
            return False
        
        # Merge all polygons from different states
        merged_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True))
        
        # Ensure the merged GDF has the same CRS as the original boundary
        merged_gdf.crs = original_crs
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the merged polygon shapefile
        merged_gdf.to_file(output_path)
        print(f"Successfully saved merged polygon shapefile to: {output_path}")
        print(f"Total polygons in output: {len(merged_gdf)}")
        print(f"Output CRS: {merged_gdf.crs}")
        
        return True
        
    except Exception as e:
        print(f"An error occurred in gssurgo_raster_to_clipped_polygon: {e}")
        return False

# # Example usage:
# if __name__ == "__main__":
#     # Example parameters
#     polygon_path = "your_boundary_polygon.shp"  # Your boundary polygon shapefile (NOT a directory)
#     base_raster_path = "/home/smdsgit/Culvert_socket/Culvert_web_app/instance/core_data/Soil_GSSURGO"  # Base directory path
#     output_path = "output/clipped_polygons.shp"  # Output shapefile path
#     usa_states_shapefile_path = "us_states.zip"  # Path to US states shapefile
    
#     # Call the function
#     success = gssurgo_raster_to_clipped_polygon(
#         polygon_path=polygon_path,
#         base_raster_path=base_raster_path,
#         output_path=output_path,
#         usa_states_shapefile_path=usa_states_shapefile_path
#     )

# -----------------------------------Function to download and process Hydro Soil Group from GSSURGO data. This function also assigns the most dominant (covering most area) soil group for the sub-watersheds.
def assign_dominant_hydrologic_soil_group_to_watersheds_using_GSSURGO(aoi_soil_data_path,
                                                                      input_watersheds_path,
                                                                  output_watersheds_path):
        # Function to convert values containing '/' to 'D'
        def convert_values(value):
            if isinstance(value, str) and '/' in value:
                return 'D'
            else:
                return value
        # Function to clip a GeoDataFrame with a polygon and find the dominant group
        def find_dominant_soil_group(aoi_soil_gdf_clipped, clip_polygon):
            gdf_clipped = gpd.clip(aoi_soil_gdf_clipped, clip_polygon)
            if gdf_clipped.empty:
                return None

            # Calculate the area of each clipped polygon
            gdf_clipped['area'] = gdf_clipped.area

            # Find the dominant group based on the largest area
            dominant_group = gdf_clipped.loc[gdf_clipped['area'].idxmax(), 'hydgrpdcd']
            return dominant_group

        aoi_soil_gdf_clipped=gpd.read_file(aoi_soil_data_path)
        # # Drop the 'geometry_x' column
        # aoi_soil_gdf_clipped = aoi_soil_gdf_clipped.drop(columns=['geometry_x'])

        # Iterate over each polygon in the shapefile and clip the GeoDataFrame
        watersheds_gdf = gpd.read_file(input_watersheds_path)
        
        for index, row in watersheds_gdf.iterrows():
            clip_polygon = row['geometry']
            dominant_group = find_dominant_soil_group(aoi_soil_gdf_clipped, clip_polygon)
            watersheds_gdf.loc[index,'HySGrp'] = dominant_group
        # print(f"before_converting {watersheds_gdf['HySGrp'].unique()}")


        # Apply the function to the 'Column2'
        watersheds_gdf['HySGrp'] = watersheds_gdf['HySGrp'].apply(convert_values)
        # print(f"after_converting {watersheds_gdf['HySGrp'].unique()}")
        watersheds_gdf['HySGrpN']=np.nan
        watersheds_gdf.loc[watersheds_gdf['HySGrp']=='A','HySGrpN']=1
        watersheds_gdf.loc[watersheds_gdf['HySGrp']=='B','HySGrpN']=2
        watersheds_gdf.loc[watersheds_gdf['HySGrp']=='C','HySGrpN']=3
        watersheds_gdf.loc[watersheds_gdf['HySGrp']=='D','HySGrpN']=4
        watersheds_gdf.to_file(output_watersheds_path,driver='ESRI Shapefile',encoding='utf-8', mode='w')

        return watersheds_gdf


# Usage
# assign_dominant_hydrologic_soil_group_to_watersheds_using_GSSURGO(zip_gSSURGO_file_path='/content/gSSURGO_SC.zip',
#                                                                   save_watershed_file_path='/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp',
#                                                                   save_gSSURGO_muaggat_soil_data_path='/content/drive/MyDrive/WS_Properties/gSSURGO_data_polygon_UTM_reprojected.shp'
#                                                                   )
# =====================================================================================================================================================================================================
# Step-1: Runoff coefficient from Table: User has neither flow nor precipitation data:
def calculate_runoff_coefficient_from_table(ws_shapefile_path = '/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp',
                                            dem_raster_path = '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/DEM_UTM_reprojected.tif',
                                            runoff_table_path = '/content/drive/MyDrive/Rational_Method/input_folder/Coeff_Runoff_table.csv',
                                            hydro_soil_shapefile_path = '/content/drive/MyDrive/WS_Properties/gSSURGO_data_polygon_UTM_reprojected.shp',
                                            temp_folder_path = '/content/temporary',
                                            output_folder = '/content/drive/MyDrive/Rational_Method/output_folder',
                                            clipped_nlcd_raster_path = None,
                                            NLCD_CONUS_raster_path = None,
                                            boundary_shapefile_path = None
                                            ):
    # Credible source for assigning NLCD values to Land Cover Types
    # https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.0/developing-a-terrain-model-and-geospatial-layers/creating-land-cover-mannings-n-values-and-impervious-layers
    # Credible source for Runoff Coefficient lookup table can be found here https://stormwater.pca.state.mn.us/runoff_coefficients_for_different_soil_groups_and_slopes
    os.makedirs(temp_folder_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Define temporary NLCD raster path and get bounding box from watershed shapefile
    temp_nlcd_raster_path = os.path.join(temp_folder_path, 'nlcd_raster.tif')
    gdf_boundary = gpd.read_file(ws_shapefile_path)

    # Check if clipped NLCD raster exists, if not create it
    if clipped_nlcd_raster_path and not os.path.exists(clipped_nlcd_raster_path):
        if NLCD_CONUS_raster_path and boundary_shapefile_path:
            # Load boundary shapefile to get CRS
            boundary_gdf = gpd.read_file(boundary_shapefile_path)
            boundary_orig_crs = boundary_gdf.crs
            
            with rasterio.Env(GDAL_CACHEMAX=64):
                with rasterio.open(NLCD_CONUS_raster_path) as src:
                    # Reproject AOI to raster CRS
                    gdf_boundary_clip = gdf_boundary.to_crs(src.crs)
                    shapes = [geom.__geo_interface__ for geom in gdf_boundary_clip.geometry]
                    
                    # Clip (only reads required blocks)
                    data, transform = rasterio.mask.mask(src, shapes, crop=True)
                    
                    # Create clean profile instead of copying
                    profile = {
                        'driver': 'GTiff',
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata,
                        'width': data.shape[2],
                        'height': data.shape[1],
                        'count': 1,
                        'crs': src.crs,
                        'transform': transform
                    }
            # Save clipped raster in original raster CRS first
            temp_clipped_path = clipped_nlcd_raster_path + '.temp.tif'
            with rasterio.open(temp_clipped_path, "w", **profile) as dst:
                dst.write(data)
            
            # Reproject to boundary CRS if different
            if profile['crs'] != boundary_orig_crs:
                with rasterio.open(temp_clipped_path) as src:
                    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                        src.crs, boundary_orig_crs, src.width, src.height, *src.bounds
                    )
                    
                    # Create clean kwargs instead of copying src.meta
                    kwargs = {
                        'driver': 'GTiff',
                        'dtype': src.dtypes[0],
                        'nodata': -9999,  # Set integer nodata value to avoid float NaN issues
                        'width': dst_width,
                        'height': dst_height,
                        'count': 1,
                        'crs': boundary_orig_crs,
                        'transform': dst_transform
                    }

                    with rasterio.open(clipped_nlcd_raster_path, "w", **kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=boundary_orig_crs,
                            resampling=Resampling.nearest,
                            src_nodata=src.nodata,
                            dst_nodata=-9999,  # Set destination nodata to -9999
                        )
                # Remove temporary file
                os.remove(temp_clipped_path)
            else:
                # Just rename if CRS is the same
                os.rename(temp_clipped_path, clipped_nlcd_raster_path)
            
            print('Status: Created clipped NLCD raster from CONUS data')
            temp_nlcd_raster_path = clipped_nlcd_raster_path
        else:
            print('Status: No fallback method - NLCD_CONUS_raster_path and boundary_shapefile_path required')
            temp_nlcd_raster_path = clipped_nlcd_raster_path
    elif clipped_nlcd_raster_path and os.path.exists(clipped_nlcd_raster_path):
        temp_nlcd_raster_path = clipped_nlcd_raster_path
        print('Status: Using existing clipped NLCD raster')
    
    # Load and reproject NLCD raster to match the watershed CRS with proper NoData handling
    nlcd = rxr.open_rasterio(temp_nlcd_raster_path, masked=True)
    
    # Get original NoData value from the raster
    original_nodata = nlcd.rio.nodata
    if original_nodata is None or pd.isna(original_nodata):
        # For Annual NLCD, the NoData value is 250 according to metadata
        original_nodata = 250.0
        print(f"Warning: No NoData value found in raster metadata, using NLCD standard: {original_nodata}")
    else:
        print(f"Status: Original NoData value from metadata: {original_nodata}")
    
    # Explicitly mask NoData values - for Annual NLCD, NoData is 250
    nlcd = nlcd.where(nlcd != 250.0)
    nlcd_reprojected = nlcd.rio.reproject(gdf_boundary.crs)
    
    # Define the path where you want to save the reprojected raster
    output_nlcd_raster_path = os.path.join(output_folder, "nlcd_UTM_reprojected.tif")
    # Save the reprojected NLCD raster with proper NoData handling
    nlcd_reprojected.rio.write_nodata(-9999, inplace=True)
    nlcd_reprojected.rio.to_raster(output_nlcd_raster_path)
    print(f"Status: Reprojected NLCD raster saved to {output_folder}")

    # Load DEM and soil shapefile
    dem = rxr.open_rasterio(dem_raster_path, masked=True)
    soil_gdf = gpd.read_file(hydro_soil_shapefile_path)
    print('Status: Loaded Soil and dem data')
    
    # Convert soil group column to numeric values and handle combined categories
    soil_gdf['hydgrpdcd'] = soil_gdf['hydgrpdcd'].replace({"A/D": "D", "B/D": "D", "C/D": "D"})
    soil_gdf['hydgrpdcd'] = soil_gdf['hydgrpdcd'].map({"A": 1, "B": 2, "C": 3, "D": 4})

    # Load runoff coefficient table and create lookup dictionary
    runoff_df = pd.read_csv(runoff_table_path)
    # Create the runoff coefficient lookup dictionary - MODIFIED to handle comma-separated NLCD values
    runoff_dict = {}
    for _, row in runoff_df.iterrows():
        # Split NLCD values by comma and strip whitespace
        nlcd_values = [int(val.strip()) for val in str(row['Land use Value NLCD']).split(',')]
        for nlcd_val in nlcd_values:
            for group in range(1, 5):  # Iterate over hydrologic groups 1 to 4
                for slope_category in ['0_2', '2_6', '6']:
                    if not pd.isna(row[f"c{group}_{slope_category}"]):
                        runoff_dict[(nlcd_val, f"c{group}_{slope_category}")] = row[f"c{group}_{slope_category}"]

    # Create an empty array to hold the combined runoff values for the entire watershed
    combined_runoff_values = np.full_like(nlcd_reprojected.values[0], -9999, dtype=np.float32)

    # Process each watershed polygon
    for index, watershed in gdf_boundary.iterrows():
        watershed_geom = [watershed['geometry']]

        # Clip reprojected NLCD raster to watershed extent
        nlcd_clipped = nlcd_reprojected.rio.clip(watershed_geom, crs=gdf_boundary.crs)
        
        # Ensure NoData values are properly handled after clipping
        nlcd_clipped = nlcd_clipped.where(nlcd_clipped != 250.0)

        # Resample DEM to match NLCD and calculate slope
        dem_resampled = dem.rio.reproject_match(nlcd_clipped, resampling=Resampling.bilinear)
        x_res, y_res = map(abs, dem_resampled.rio.resolution())
        dem_data = dem_resampled.values[0]
        slope_percentage = np.hypot(*np.gradient(dem_data, y_res, x_res)) * 100

        # Rasterize soil data to match NLCD raster transform and extent
        soil_raster = rasterize(
            [(mapping(geom), value) for geom, value in zip(soil_gdf.geometry, soil_gdf['hydgrpdcd'])],
            out_shape=nlcd_clipped.shape[1:],
            transform=nlcd_clipped.rio.transform(),
            fill=np.nan,
            dtype=np.float32
        )
        soil_raster_xarray = xr.DataArray(
            soil_raster, dims=["y", "x"],
            coords={"y": nlcd_clipped.y, "x": nlcd_clipped.x},
            attrs=nlcd_clipped.attrs
        )

        # Categorize slope for runoff table lookup
        slope_categories = np.where(slope_percentage < 2, '0_2', np.where(slope_percentage < 6, '2_6', '6'))

        # Enhanced lookup function to handle NoData values properly
        def lookup_runoff(nlcd_val, soil_val, slope_cat):
            # Check for both NaN and NoData values (250.0 for Annual NLCD)
            if (np.isnan(nlcd_val) or np.isnan(soil_val) or 
                nlcd_val == 250.0 or nlcd_val == -9999):
                return np.nan
            
            # Convert to integer for lookup
            try:
                nlcd_int = int(nlcd_val)
                soil_int = int(soil_val)
            except (ValueError, TypeError):
                return np.nan
            
            # Check if combination exists in lookup table
            lookup_key = (nlcd_int, f"c{soil_int}_{slope_cat}")
            if lookup_key not in runoff_dict:
                print(f"Warning: No runoff coefficient found for NLCD={nlcd_int}, Soil Group={soil_int}, Slope={slope_cat}")
                return np.nan
                
            return runoff_dict[lookup_key]

        runoff_values = np.vectorize(lookup_runoff)(nlcd_clipped.values[0], soil_raster, slope_categories)

        # Calculate pixel area
        pixel_width, pixel_height = nlcd_clipped.rio.resolution()
        pixel_area = abs(pixel_width * pixel_height)

        # Flatten arrays and create comprehensive mask
        runoff_values_flat = runoff_values.flatten()
        nlcd_flat = nlcd_clipped.values[0].flatten()
        soil_flat = soil_raster.flatten()

        # Create comprehensive mask excluding NaNs and NoData values
        mask = (~np.isnan(runoff_values_flat) & 
                ~np.isnan(nlcd_flat) & 
                ~np.isnan(soil_flat) &
                (nlcd_flat != 255.0) &
                (nlcd_flat != -9999) &
                (runoff_values_flat != -9999))

        if np.sum(mask) == 0:
            print(f"Warning: No valid pixels found for watershed {index}")
            gdf_boundary.loc[index, 'C_Runoff'] = np.nan
            continue

        # Calculate the total area by counting only valid pixels
        total_area = np.sum(mask) * pixel_area

        # Perform weighted average calculation
        weighted_sum = np.sum(runoff_values_flat[mask] * pixel_area)
        weighted_average_runoff = weighted_sum / total_area

        # Add the area weighted runoff coefficient to the watershed polygons gdf
        gdf_boundary.loc[index, 'C_Runoff'] = weighted_average_runoff

        print(f"Status: Processed watershed {index}, valid pixels: {np.sum(mask)}, runoff coeff: {weighted_average_runoff:.4f}")

        # Update the combined runoff raster with proper masking
        # Get the bounds of the current watershed in the full raster coordinate system
        watershed_bounds = nlcd_clipped.rio.bounds()
        full_bounds = nlcd_reprojected.rio.bounds()
        
        # Handle bounds whether they're tuples or BoundingBox objects
        if hasattr(watershed_bounds, 'left'):
            w_left, w_bottom, w_right, w_top = watershed_bounds.left, watershed_bounds.bottom, watershed_bounds.right, watershed_bounds.top
        else:
            w_left, w_bottom, w_right, w_top = watershed_bounds
            
        if hasattr(full_bounds, 'left'):
            f_left, f_bottom, f_right, f_top = full_bounds.left, full_bounds.bottom, full_bounds.right, full_bounds.top
        else:
            f_left, f_bottom, f_right, f_top = full_bounds
        
        # Calculate indices for placing watershed data in the full raster
        x_res, y_res = nlcd_reprojected.rio.resolution()
        col_start = int((w_left - f_left) / abs(x_res))
        col_end = col_start + nlcd_clipped.shape[2]
        row_start = int((f_top - w_top) / abs(y_res))
        row_end = row_start + nlcd_clipped.shape[1]
        
        # Ensure indices are within bounds
        col_start = max(0, col_start)
        col_end = min(combined_runoff_values.shape[1], col_end)
        row_start = max(0, row_start)
        row_end = min(combined_runoff_values.shape[0], row_end)
        
        # Only update pixels that have valid runoff values
        valid_mask = ~np.isnan(runoff_values)
        if col_end > col_start and row_end > row_start:
            # Adjust runoff_values size to match the target slice
            target_rows = row_end - row_start
            target_cols = col_end - col_start
            if runoff_values.shape[0] != target_rows or runoff_values.shape[1] != target_cols:
                # Resize if needed
                runoff_values_resized = runoff_values[:target_rows, :target_cols]
                valid_mask_resized = valid_mask[:target_rows, :target_cols]
            else:
                runoff_values_resized = runoff_values
                valid_mask_resized = valid_mask
                
            combined_runoff_values[row_start:row_end, col_start:col_end][valid_mask_resized] = runoff_values_resized[valid_mask_resized]

    print('Status: Calculated Area weighted average Runoff coefficient for all watersheds')
    
    # Create output raster with proper NoData handling
    combined_runoff_values_reshaped = combined_runoff_values[np.newaxis, :, :]
    combined_runoff_xarray = nlcd_reprojected.copy(data=combined_runoff_values_reshaped)
    
    # Set NoData value for output raster
    combined_runoff_xarray.rio.write_nodata(-9999, inplace=True)
    # Mask areas where original NLCD was NoData (255.0)
    combined_runoff_xarray = combined_runoff_xarray.where((nlcd_reprojected != 255.0) & ~nlcd_reprojected.isnull(), -9999)
    
    output_raster_path = os.path.join(output_folder, "Runoff_coeff_watersheds_from_Table_UTM.tif")
    combined_runoff_xarray.rio.to_raster(output_raster_path)

    print(f"Status: Runoff coefficient raster saved to {output_folder}")
    
    output_shapefile_path = os.path.join(output_folder, "CRunoff_per_watershed_UTM_reprojected.shp")
    gdf_boundary.to_file(output_shapefile_path)
    print(f"Status: final watershed polygon shapefile saved to {output_folder}")
    
    # Print summary statistics
    valid_watersheds = gdf_boundary['C_Runoff'].notna()
    if valid_watersheds.any():
        print(f"Summary: {valid_watersheds.sum()} watersheds processed successfully")
        print(f"Runoff coefficient range: {gdf_boundary['C_Runoff'].min():.4f} - {gdf_boundary['C_Runoff'].max():.4f}")
        print(f"Mean runoff coefficient: {gdf_boundary['C_Runoff'].mean():.4f}")
    else:
        print("Warning: No watersheds processed successfully")
    
    return gdf_boundary

#==========================================================================================================================================
# ************************* BACK - CALCULATION OF C-RUNOFF
#==============================================================================================================================
"""
        ## Calculating Coefficient of Runoff using Rational Method
    > The Rational Method, a simplified approach to estimating peak runoff rates, is based on several key assumptions:
    1. Uniform Rainfall Intensity: The rainfall intensity is assumed to be uniform over the entire watershed for a duration equal to or greater than the time of concentration.
    2. Time of Concentration: The time of concentration (Tc) is the time required for runoff from the most remote point in the watershed to reach the outlet. It is assumed to be constant and independent of rainfall intensity.
    3. Negligible Storage Effects: The effects of storage in the watershed, such as infiltration and depression storage, are assumed to be negligible.
    4. Invariant Runoff Coefficient: The runoff coefficient (C) is assumed to be constant and independent of rainfall intensity and antecedent moisture conditions.
    5. Small Watershed: The Rational Method is best suited for small watersheds, typically less than 200 acres (80 hectares). For larger watersheds, more complex methods may be necessary.
    6. Overland Flow Dominance: The method assumes that overland flow is the dominant flow path, neglecting subsurface flow and channel flow.
    7. Constant Rainfall Duration: The rainfall duration is assumed to be equal to the time of concentration.

    ```
                        C = Q/IA
                            where:
                            C = Runoff coefficient (dimensionless).
                            Q = Peak flow, ft3/s.                        
                            I = Rainfall intensity, in/hr.
                            A = Drainage area, acres.
    ```

"""


def back_calculate_Coeff_of_Runoff(flow_file_path, precip_file_path, select_top=10000):
    """
    Calculate coefficient of runoff from flow and precipitation data
    without using datetime functions.

    Parameters:
    flow_file_path (str): Path to flow data CSV file
    precip_file_path (str): Path to precipitation data CSV file
    select_top (int): Number of top flow values to consider

    Returns:
    float: Mean coefficient of runoff value
    """
    # Load data
    f = pd.read_csv(flow_file_path)
    pr = pd.read_csv(precip_file_path)

    # Fill missing Hr values with 0 in both datasets
    f['Hr'] = f['Hr'].fillna(0)
    f['Min'] = f['Min'].fillna(0)
    pr['Hr'] = pr['Hr'].fillna(0)
    pr['Min'] = pr['Min'].fillna(0)

    # Merge dataframes on common columns (Year, Month, Day, Hr, Min)
    df = pd.merge(f, pr[['Year', 'Month', 'Day', 'Hr', 'Min', 'PI']], 
                 on=['Year', 'Month', 'Day', 'Hr', 'Min'],
                 how='left')
    # Adjust select_top if df is smaller
    if len(df) < select_top:
        select_top = int(0.3 * len(df))  # 30% of df length
    
    # Check for missing values in Year, Month, Day columns individually
    if df['Year'].isna().all() or df['Month'].isna().all() or df['Day'].isna().all():
        raise ValueError("Upload valid data, Year, Month, Day columns are currently missing.")
    
    df = df.dropna(subset=['Year'])
    
    # Fill missing PI values with 0
    df['PI'] = df['PI'].fillna(0)

    # Create dry period identification
    df['is_dry'] = df['PI'] == 0

    # Create groups of consecutive dry/wet periods
    df['group'] = (df['is_dry'] != df['is_dry'].shift()).cumsum()

    # Filter for start of wet periods after dry periods of 7 or more hours
    result_df = df[
        (df['group'].shift(1) != df['group']) &  # Start of new period
        (df['is_dry'] == False) &                # Wet period
        (df.groupby('group')['is_dry'].transform('count').shift(1) >= 7)  # Previous dry period ≥ 7 hours
    ]

    # Clean up and prepare final dataset
    result_df = result_df.drop(columns=['is_dry', 'group'])
    result_df = result_df[~result_df['Flow'].isna()]
    result_df = result_df.sort_values(by='Flow', ascending=False)
    result_df = result_df.head(select_top)

    # Calculate coefficient of runoff
    result_df['Cval'] = ((result_df['Flow']/result_df['Area_km2'])* 0.143) / (result_df['PI'] / 2.54)

    # Filter out invalid values and calculate mean
    Cvals = result_df['Cval'][~result_df['Cval'].isin([np.nan, np.inf, -np.inf])]

    return Cvals.mean()


def back_calculate_runoff_for_gauged_WSs(flow_file_dir, precip_file_dir,single_site_shapefile_path, ws_char_path, roi_df_dir_path, CRunoff_output_dir_path,Gst_Names, select_top=10000):
    """
    Automate the process of calculating coefficient of runoff for multiple Gst_Names.
    
    Parameters:
    Gst_Names (list): List of Gst Names (e.g., ['WS77', 'WS78', 'WS79'])
    flow_file_dir (str): Directory where flow files are stored
    precip_file_dir (str): Directory where precipitation files are stored
    select_top (int): Number of top flow values to consider
    
    Returns:
    pd.DataFrame: DataFrame with Gst_ID and their respective MeanCvalue
    """
    cval_data = {'Gst_ID': [], 'MeanCvalue': []}

    for gst in Gst_Names:
        # Dynamically generate file paths based on Gst_Name
        flow_file_path = os.path.join(flow_file_dir, f"full_stream_series_{gst}.csv")
        precip_file_path = os.path.join(precip_file_dir, f"full_precip_series_{gst}.csv")        
        try:
            mean_cval = back_calculate_Coeff_of_Runoff(flow_file_path, precip_file_path, select_top)
            print(mean_cval)
            cval_data['Gst_ID'].append(gst)
            cval_data['MeanCvalue'].append(mean_cval)
        except Exception as e:
            print(f"Error processing {gst}: {e}")
            cval_data['Gst_ID'].append(gst)
            cval_data['MeanCvalue'].append(None)

    # Create DataFrame with calculated values
    cval_df = pd.DataFrame(cval_data)
    
    if len(Gst_Names)==1:
        gdf = gpd.read_file(single_site_shapefile_path)
        gdf['C_Runoff'] = cval_df['MeanCvalue'].iloc[0]
    else:
        roi_df_path = os.path.join(roi_df_dir_path,'roi_index.csv')
        roi_df = pd.read_csv(roi_df_path)
        gdf = gpd.read_file(ws_char_path)


        # Ensure column names in roi_df are treated as strings for matching
        roi_df.columns = roi_df.columns.astype(int)
        roi_df.columns
        # Convert gdf['Point_ID'] to string for consistency
        gdf['Point_ID'] = gdf['Point_ID'].astype(int)

        # Map the first row of roi_df to the matching Point_ID in gdf
        gdf['ROI'] = gdf['Point_ID'].map(roi_df.iloc[0])
        
        gdf['C_Runoff'] = gdf['GWS_ID'].map(cval_df.set_index('Gst_ID')['MeanCvalue'])

        import numpy as np

        # Step 1: Filter rows where C_Runoff is not NA
        valid_c_runoff = gdf.loc[~gdf['C_Runoff'].isna(), ['Point_ID', 'C_Runoff']]

        # Step 2: Create a mapping of Point_ID to C_Runoff
        point_to_runoff_map = dict(zip(valid_c_runoff['Point_ID'], valid_c_runoff['C_Runoff']))

        # Step 3: Fill NA values in C_Runoff using the ROI-to-Point_ID mapping
        gdf.loc[gdf['C_Runoff'].isna(), 'C_Runoff'] = gdf['ROI'].map(point_to_runoff_map)

    output_shapefile_path=os.path.join(CRunoff_output_dir_path, "CRunoff_per_watershed_UTM_reprojected.shp") 
    gdf.to_file(output_shapefile_path)   
    return gdf
#===========================================================================================================================================
# ======Function to calculate specific peak discharge using the point PIDFs from NOAA atlas14 ========================================================================================================================
"""
Step-2: Calculating specific discharge from NOAA Atlas PIDFs: When user has no precipitation data:

Important Notes

    It should be noted that precipitation frequency estimates from NOAA Atlas 14 are point estimates and are not directly applicable to larger areas. 
    The conversion of a point to an areal estimate is usually done by applying an appropriate areal reduction factor to the average of the point estimates within the subject area.
    Areal reduction factors are generally a function of the size of an area and the duration of the precipitation. 
    The depth-area-duration curves from the Technical Paper No. 29 (U.S. Weather Bureau, 1957), developed for the contiguous United States, can be used for this purpose. (read more)! 
    The noaa data can be downloaded from here

    Area Reduction factor is derived from the Technical Paper 29 (U.S. Weather Bureau, 1957) as described here.
"""

# Function to calculate specific peak discharge using the point PIDFs from NOAA atlas14
def Calculate_peakQ_using_RM_from_NOAA_Atlas(crunoff_shapefile_dir_path,
                               RM_output_dir_path,
                               temporary_folder,
                               pointPI=None,
                               RP_list=None,
                               ts_type='ams'):

        # Check if RP_list length exceeds 10
        if len(RP_list) > 10:
            raise ValueError("Error: RP_list length exceeds 10.")

        # Check if any RP_list values are greater than 1000 or less than/equal to 1
        if any(x > 1000 for x in RP_list) or any(x <= 1 for x in RP_list):
            raise ValueError("Error: RP_list values must be between 1 and 1000.")

        ####################################################################################################
        ####### Create directory if it doesn't exist
        #####################################################################################################
        ### Create directory for temporary folder
        os.makedirs(temporary_folder, exist_ok=True)
        # Create directory for saving the noaa pidfs
        pidf_files_path=os.path.join(temporary_folder,'pidfs')
        os.makedirs(pidf_files_path, exist_ok=True)

        ###################################################################################################
        ######## Extracting Return values from NOAA-Atlas data for given storm durations and return intervals
        #################################################################################################
        def extract_PI_from_NOAA(file_path, dur, ri):
            def interpolate_return_level(ri, rl_list, rv_list):
                # Check if ri exists in rl_list
                if ri in rl_list:
                    return rv_list[rl_list.index(ri)]
                else:
                    idx = np.searchsorted(rl_list, ri)
                    if idx == 0:
                        return rv_list[0]
                    elif idx == len(rl_list):
                        return rv_list[-1]
                    else:
                        x0, x1 = rl_list[idx-1], rl_list[idx]
                        y0, y1 = rv_list[idx-1], rv_list[idx]
                        return y0 + (y1 - y0) * ((ri - x0) / (x1 - x0))

            # Read file lines
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Fetch line 14 for column names
            column_names_line = lines[13]
            column_names = [int(val.split('/')[1]) for val in column_names_line.replace("by duration for AEP:,", "").replace("'", "").strip().split(',')]

            # Return periods directly available in NOAA Atlas data
            rl_list = [2, 5, 10, 25, 50, 100, 200, 500, 1000]

            # Create list to store DataFrames
            dfs = []

            # PRECIPITATION FREQUENCY ESTIMATES
            data, durations = [], []
            for line in lines[14:33]:
                duration = line.split(':')[0]
                durations.append(duration)
                data.append([float(val) for val in line.split(':')[1].strip().split(',')[1:]])

            df1 = pd.DataFrame(data, index=durations, columns=column_names)
            dfs.append(df1)

            # PRECIPITATION FREQUENCY ESTIMATES AT UPPER BOUND OF 90% CONFIDENCE INTERVAL
            data, durations = [], []
            for line in lines[36:55]:
                duration = line.split(':')[0]
                durations.append(duration)
                data.append([float(val) for val in line.split(':')[1].strip().split(',')[1:]])

            df2 = pd.DataFrame(data, index=durations, columns=column_names)
            dfs.append(df2)

            # PRECIPITATION FREQUENCY ESTIMATES AT LOWER BOUND OF 90% CONFIDENCE INTERVAL
            data, durations = [], []
            for line in lines[58:77]:
                duration = line.split(':')[0]
                durations.append(duration)
                data.append([float(val) for val in line.split(':')[1].strip().split(',')[1:]])

            df3 = pd.DataFrame(data, index=durations, columns=column_names)
            dfs.append(df3)

            # Error handling in case dur is not in index
            if dur not in dfs[0].index:
                raise ValueError(f"The specified duration '{dur}' was not found in the data.")

            # Interpolate return values
            rv_low = interpolate_return_level(ri, rl_list, dfs[2].loc[dur].tolist())
            rv_est = interpolate_return_level(ri, rl_list, dfs[0].loc[dur].tolist())
            rv_upr = interpolate_return_level(ri, rl_list, dfs[1].loc[dur].tolist())

            rv = [rv_low, rv_est, rv_upr]
            return rv


        ###############################################################################################
        ###### Fucntion to find the closest duration match
        ###############################################################################################
        def find_closest_duration(durgdf):
          # Issue warning if closest duration exceeds 24 hours
          if durgdf > 1440:
              print(f"Warning: duration exceeds 24 hours: Rational Method and TP-29 ARF are not suitable method for this watershed")

          # Create DataFrame with durations
          durations_df = pd.DataFrame({
              'Duration': ['5-min', '10-min', '15-min', '30-min', '60-min',
                          '2-hr', '3-hr', '6-hr', '12-hr', '24-hr',
                          '2-day', '3-day', '4-day', '7-day', '10-day',
                          '20-day', '30-day', '45-day', '60-day']
          })

          # Convert durations to minutes
          durations_min = durations_df['Duration'].apply(lambda x:
              float(x.replace('min', '').replace('-', '')) if 'min' in x
              else float(x.replace('hr', '').replace('-', '')) * 60
              if 'hr' in x
              else float(x.replace('day', '').replace('-', '')) * 1440)

          # Find closest duration
          closest_duration_idx = np.argmin(np.abs(durations_min - durgdf))
          closest_duration_match = durations_df.loc[closest_duration_idx, 'Duration']
          closest_duration_min = durations_min[closest_duration_idx]
          return closest_duration_match, closest_duration_min

        ############################################################################################
        ###### Function to calculate the area reduction factor for area follwoing Bell, 1976 https://nora.nerc.ac.uk/id/eprint/5751/ 
        ############################################################################################
        def calculate_ARF(dur, area_ha):
          ARF = 1 - np.exp(-1.1*float(dur)**0.25) + np.exp(-1.1*float(dur)**0.25 - 2.59*10**-2*float(area_ha)*0.01)
          return ARF

        ##############################################################################################################################
        # Load ws polygon shapefile
        input_shapefile_path=os.path.join(crunoff_shapefile_dir_path, "CRunoff_per_watershed_UTM_reprojected.shp")
        gdf_input = gpd.read_file(input_shapefile_path).to_crs('EPSG:4326')
        # Merge (dissolve) the polygons based on the 'Point_ID'
        gdf = gdf_input.dissolve(by='Point_ID')
        # reset the index to make the 'Point_ID' a regular column
        gdf = gdf.reset_index()
    
        pidf_df=pd.DataFrame()
        q_df = pd.DataFrame()
        # Iterate through each polygon
        for index, row in gdf.iterrows():
              durgdf = row['TCmin']
              dur, closest_duration_min = find_closest_duration(durgdf)
              # Get polygon geometry
              polygon = row.geometry
              # Check if geometry is a Polygon or MultiPolygon
              if polygon.geom_type == 'Polygon':
                  centroid = polygon.centroid
              elif polygon.geom_type == 'MultiPolygon':
                  # Calculate weighted centroid for MultiPolygon
                  centroid = polygon.centroid
              # Extract longitude and latitude
              lon = centroid.x
              lat = centroid.y
              # print(f"Polygon {index+1} Centroid: Lon={lon}, Lat={lat}")
              
              # Initialize empty DataFrame
              rvdf = pd.DataFrame()
              pidf = pd.DataFrame()
              
              if pointPI is None:
                  # read the docs about this NOAA-Atlas14 api https://www.weather.gov/media/owp/hdsc_documents/NA14_Sec5_PFDS.pdf
                  url=f'https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/fe_text.csv?lat={lat}&lon={lon}&data=intensity&units=english&series={ts_type}'

                  file_path = os.path.join(pidf_files_path,f'PI_{lon}_{lat}_{ts_type}_inch_per_hr.txt')
                  download_data(url, file_path,verbose=False)
                  
                  for ind, rvrow in enumerate(RP_list):
                      ri = rvrow
                      rvd = extract_PI_from_NOAA(file_path, dur, ri)

                      # Assign list elements to new columns
                      rvdf[f'RM{ri}yrL'] = [rvd[0]]
                      rvdf[f'RM{ri}yrE'] = [rvd[1]]
                      rvdf[f'RM{ri}yrU'] = [rvd[2]]

                      # Assign list elements to new columns
                      pidf[f'PI{ri}yrL'] = [rvd[0]]
                      pidf[f'PI{ri}yrE'] = [rvd[1]]
                      pidf[f'PI{ri}yrU'] = [rvd[2]]

                  # Determine correction factor based on the return interval (ri)
                  if ri == 25:
                    Correction_factor = 1.1
                  elif ri == 50:
                    Correction_factor = 1.2
                  elif ri == 100:
                    Correction_factor = 1.25
                  else:
                    Correction_factor = 1.0
              else:
                  # Use pointPI data (already in in/hr)
                  rvdf[f'RMevent'] = [float(pointPI)]
                  pidf[f'PIevent'] = [float(pointPI) * 2.54]  # converting from inch/hr to cm/hr
                  Correction_factor = 1.0  # No correction factor for pointPI
                
              
              ARF = calculate_ARF(closest_duration_min/60, row['area_ha'])
              rvdf = rvdf * ARF       ## areal estimate but unit is still in/hr, which is used in Q = CIA
              pidf = pidf * ARF*2.54 if pointPI is None else pidf * ARF  ## areal estimate and then converting from inch/hr to cm/hr
              
              area=row['area_ha']*2.47105 # converting from ha to acre 
              Crunoff=row['C_Runoff']
              # Append rvdf to rv_df with the index of the current row
              rvdf.index = [index]        ## Set the index to match the current row index in gdf
              pidf.index = [index]        ## Set the index to match the current row index in gdf
              pidf_df = pd.concat([pidf_df, pidf], axis=0)  # Append the data for this row
              qv = (Correction_factor*Crunoff*rvdf*area*0.02831683199881)
              q_df = pd.concat([q_df, round(qv,2)], axis=0)  #m3/s/km2

        # Concatenate pidf_df with gdf and return the updated gdf
        gdf_pidf = gdf.join(pidf_df, how='outer')  # Outer join to ensure all rows in gdf are preserved
        output_pidf_shapefile_path = os.path.join(RM_output_dir_path, "PIDF_cmperhr_per_watershed_UTM_reprojected.shp")
        gdf_pidf.to_file(output_pidf_shapefile_path)


        gdf_peakQ= gdf.join(q_df, how='outer')
        output_qdf_shapefile_path = os.path.join(RM_output_dir_path, "RMQ_m3perSec_per_watershed_UTM_reprojected.shp")
        gdf_peakQ.to_file(output_qdf_shapefile_path)



        print(f"Status: Rational Method based discharges in m3/s for {RP_list} year saved to {RM_output_dir_path}")
        return gdf_peakQ
# Usage

# ws_gdf=Calculate_peakQ_using_RM_from_NOAA_Atlas(input_shapefile_path='/content/drive/MyDrive/Rational_Method/output_folder/PIDF_cmperhr_per_watershed_UTM_reprojected.shp',
#                                RP_list=[25,50,100],
#                                output_folder='/content/drive/MyDrive/Rational_Method/output_folder',
#                                temporary_folder='/content/temporary',
#                                ts_type='ams')


# # ======================================================================================================
# # Function to calculate specific peak discharge using the point PIDFs from NOAA atlas14
# def Calculate_peakQ_using_RM_PI_from_WS(crunoff_output_folder,
#                                            input_pidf_csv_path, 
#                                            save_output_gdf_folder_path):
    
#     # Read the GeoDataFrame from the shapefile
#     gdf_shapefile_path=os.path.join(crunoff_output_folder, "CRunoff_per_watershed_UTM_reprojected.shp")
#     gdf = gpd.read_file(gdf_shapefile_path)
#     # Merge (dissolve) the polygons based on the 'Point_ID'
#     gdf_merged = gdf.dissolve(by='Point_ID')
#     # reset the index to make the 'Point_ID' a regular column
#     gdf_merged = gdf_merged.reset_index()
    
    
#     gdf_pidf = gpd.read_file(gdf_shapefile_path)
#     # Merge (dissolve) the polygons based on the 'Point_ID'
#     gdf_merged_pidf = gdf_pidf.dissolve(by='Point_ID')
#     # reset the index to make the 'Point_ID' a regular column
#     gdf_merged_pidf = gdf_merged_pidf.reset_index()
    
#     # Columns of interest in gdf_merged
#     columns_of_interest = ["area_ha", "GWS_ID", "C_Runoff", "TCmin"]

#     if not all(col in gdf_merged.columns for col in columns_of_interest):
#         raise ValueError("One or more required columns ('area_ha', 'GWS_ID', 'C_Runoff', 'TCmin') are missing in the GeoDataFrame.")

#     # ARF calculation function using provided formula
#     def calculate_ARF(dur, area_ha):
#         ARF = 1 - np.exp(-1.1 * float(dur)**0.25) + np.exp(-1.1 * float(dur)**0.25 - 2.59*10**-2 * float(area_ha) * 0.01)
#         return ARF

#     # Conversion factor: converts I from cm/hr to in/hr, A from ha to acres, then ft³/s to m³/s.
#     conversion_factor = 0.393701 * 2.47105 * 0.0283168  # ≈ 0.027514

#     try:
#         # Read the Excel sheet into a DataFrame
#         df = pd.read_excel(input_pidf_csv_path)
#         required_columns_excel = ['Estimate', 'Lower_CI', 'Upper_CI', 'Return_Period']
#         if not all(col in df.columns for col in required_columns_excel):
#             raise ValueError(f"One or more required columns {required_columns_excel} are missing in the Excel file")
#         print("Successfully read the Excel file")
        
#         # Process each return period value from Excel
#         for _, row in df.iterrows():
#             # Round the Return_Period to use in column names
#             rp_value = round(row['Return_Period'])
#             estimate_col = f"RM{rp_value}yrE"
#             lower_ci_col = f"RM{rp_value}yrL"
#             upper_ci_col = f"RM{rp_value}yrU"
            
#             pidf_estimate_col = f"PI{rp_value}yrE"
#             pidf_lower_ci_col = f"PI{rp_value}yrL"
#             pidf_upper_ci_col = f"PI{rp_value}yrU"
               
#             # Loop through each row in the GeoDataFrame
#             for index, gdf_merged_row in gdf_merged.iterrows():
#                 area = gdf_merged_row['area_ha']
#                 runoff_coeff = gdf_merged_row['C_Runoff']
#                 closest_duration_min = gdf_merged_row['TCmin']
                
#                 # Determine the adjusted rainfall intensity (rvdf) in cm/hr.
#                 # For areas ≤400 ha, no ARF is applied.
#                 if area <= 400:
#                     ARF = 1.0
#                     rvdf = row['Estimate']
#                     lower_ci_val = row['Lower_CI']
#                     upper_ci_val = row['Upper_CI']
                    
#                 elif 400 < area < 110000:
#                     ARF = calculate_ARF(closest_duration_min / 60, area)  # duration in hours
#                     rvdf = row['Estimate'] * ARF
#                     lower_ci_val = row['Lower_CI'] * ARF
#                     upper_ci_val = row['Upper_CI'] * ARF
#                 else:
#                     rvdf = np.nan
#                     lower_ci_val = np.nan
#                     upper_ci_val = np.nan
#                     print(f"NaN values inserted for watershed area greater than 110000 Ha at index {index}")
                
#                 # Now compute Q:
#                 # Q = C * I * A, where:
#                 # I is the adjusted rainfall intensity in in/hr (I in cm/hr × 0.393701),
#                 # A is the drainage area in acres (area in ha × 2.47105),
#                 # and Q (ft³/s) is converted to m³/s by multiplying by 0.0283168.
#                 # The combined conversion factor is precomputed as conversion_factor.
#                 Q_est = runoff_coeff * rvdf * area * conversion_factor
#                 Q_lower = runoff_coeff * lower_ci_val * area * conversion_factor
#                 Q_upper = runoff_coeff * upper_ci_val * area * conversion_factor
                
#                 # Store the computed Q values in the GeoDataFrame.
#                 gdf_merged.at[index, estimate_col] = round(Q_est, 2) if isinstance(Q_est, (int, float)) and not np.isnan(Q_est) else Q_est
#                 gdf_merged.at[index, lower_ci_col] = round(Q_lower, 2) if isinstance(Q_lower, (int, float)) and not np.isnan(Q_lower) else Q_lower
#                 gdf_merged.at[index, upper_ci_col] = round(Q_upper, 2) if isinstance(Q_upper, (int, float)) and not np.isnan(Q_upper) else Q_upper

                
#                 # Assigning PIDF values to each column 
#                 pidf = row['Estimate']
#                 pidf_lower_ci_val = row['Lower_CI']
#                 pidf_upper_ci_val = row['Upper_CI']
                    
#                 # Store the computed PIDF values in the GeoDataFrame.
#                 gdf_merged_pidf.at[index, pidf_estimate_col] = round(pidf, 2) 
#                 gdf_merged_pidf.at[index, pidf_lower_ci_col] = round(pidf_lower_ci_val, 2) 
#                 gdf_merged_pidf.at[index, pidf_upper_ci_col] = round(pidf_upper_ci_val, 2) 
        
#         # Save updated shapefile
#         output_pidf_shapefile_path = os.path.join(crunoff_output_folder, "PIDF_cmperhr_per_watershed_UTM_reprojected.shp")
#         gdf_merged_pidf.to_file(output_pidf_shapefile_path)  
                
#     except Exception as e:
#         print(f"Error processing Excel file: {e}")
#     output_qdf_shapefile_path = os.path.join(save_output_gdf_folder_path, "RMQ_m3perSec_per_watershed_UTM_reprojected.shp")
#     gdf_merged.to_file(output_qdf_shapefile_path)
#     return gdf_merged


# ======================================================================================================
# Function to calculate specific peak discharge using the point PIDFs from NOAA atlas14
def Calculate_peakQ_using_RM_PI_from_WS(crunoff_output_folder,
                                           input_pidf_output_dir_path, 
                                           save_output_gdf_folder_path):
    
    # Read the GeoDataFrame from the shapefile
    gdf_shapefile_path=os.path.join(crunoff_output_folder, "CRunoff_per_watershed_UTM_reprojected.shp")
    gdf = gpd.read_file(gdf_shapefile_path)
    # Convert 'Point_ID' to integer
    gdf['Point_ID'] = gdf['Point_ID'].astype(int)

    # Merge (dissolve) polygons based on 'Point_ID'
    gdf_merged = gdf.dissolve(by='Point_ID').reset_index()

    # Columns of interest
    columns_of_interest = ["area_ha", "GWS_ID", "C_Runoff", "TCmin"]

    # Ensure all required columns exist
    missing_cols = [col for col in columns_of_interest if col not in gdf_merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in the GeoDataFrame: {missing_cols}")
    input_pidf_shapefile_path = os.path.join(input_pidf_output_dir_path,"PIDF_cmperhr_per_watershed_UTM_reprojected.shp")
    gdf_pidf = gpd.read_file(input_pidf_shapefile_path)
    gdf_pidf['Point_ID'] = gdf_pidf['Point_ID'].astype(int)

    # Merge (dissolve) polygons based on 'Point_ID'
    gdf_merged_pidf = gdf_pidf.dissolve(by='Point_ID').reset_index()

    # Replace 'C_Runoff' in gdf_merged_pidf with values from gdf_merged
    if 'C_Runoff' in gdf_merged_pidf.columns:
        gdf_merged_pidf.drop(columns=['C_Runoff'], inplace=True)  # Remove existing column to avoid conflicts

    gdf_merged_pidf = gdf_merged_pidf.merge(
        gdf_merged[['Point_ID', 'C_Runoff']], on='Point_ID', how='left'
    )
    # Identify columns that start with 'PI'
    pi_columns = [col for col in gdf_merged_pidf.columns if col.startswith('PI')]

    # Rename columns to replace 'PI' with 'RM'
    gdf_merged_pidf.rename(columns={col: col.replace('PI', 'RM') for col in pi_columns}, inplace=True)
    
    # Identify columns that start with 'RM'
    rm_columns = [col for col in gdf_merged_pidf.columns if col.startswith('RM')]
    # Now compute Q:
    # Q = C * I * A, where:
    # I is the adjusted rainfall intensity in in/hr (I in cm/hr × 0.393701),
    # A is the drainage area in acres (area in ha × 2.47105),
    # and Q (ft³/s) is converted to m³/s by multiplying by 0.0283168.
    # The combined conversion factor is precomputed as conversion_factor.
    
    # Conversion factor: converts I from cm/hr to in/hr, A from ha to acres, then ft³/s to m³/s.
    conversion_factor = 0.393701 * 2.47105 * 0.0283168  # ≈ 0.027514
    area = gdf_merged_pidf['area_ha']
    runoff_coeff = gdf_merged_pidf['C_Runoff']
    fa = area * runoff_coeff * conversion_factor
    
    # Multiply each row of fa with the corresponding row in rm_columns
    gdf_merged_pidf[rm_columns] = gdf_merged_pidf[rm_columns].mul(fa, axis=0)

    # Save the resulting GeoDataFrame to Q_gdf
    Q_gdf = gdf_merged_pidf.copy()
    
    output_qdf_shapefile_path = os.path.join(save_output_gdf_folder_path, "RMQ_m3perSec_per_watershed_UTM_reprojected.shp")
    Q_gdf.to_file(output_qdf_shapefile_path)
    return Q_gdf

# =======================================================================================================
def add_layers_to_basemap(base_map, result_gdf, name):
    def random_color():
        """Generates a random hex color code."""
        r = lambda: random.randint(0, 255)
        return '#%02X%02X%02X' % (r(), r(), r())

    gdf_to_plot = result_gdf.to_crs("EPSG:4326")

    if gdf_to_plot.shape[0] >= 1:
        # Create a FeatureGroup for the watershed polygons
        watershed_group = folium.FeatureGroup(name=name, show=True)

        # Loop through each geometry in the GeoDataFrame
        for _, row in gdf_to_plot.iterrows():
            # Skip non-polygon geometries
            if row.geometry.geom_type not in ['Polygon', 'MultiPolygon']:
                continue

            # Extract attributes with fallback to 'NA' for missing values
            point_id = row.get('Point_ID', 'NA')
            culvert_id = row.get('Point_Name', 'NA')
            WS_id = row.get('GWS_ID','NA')
            culvert_shape = row.get('PourSha', 'NA')
            material = row.get('Material', 'NA')
            width = row.get('Width_ft', 'NA') 
            height = row.get('Height_ft', 'NA') 
            flag_gauging_station = row.get('Flag_Gst', 'NA')
            paired_group_ID = row.get('Grp_ID', 'NA')
            paired_group_size = row.get('Grp_Size', 'NA')
            avg_sl_percentage = row.get('AvgSL', 'NA')
            channel_length_m = row.get('ChLen_m', 'NA')
            overflow_length_m = row.get('OvLen_m', 'NA')
            time_of_overflow_min = row.get('TOVmin', 'NA')
            time_of_channel_min = row.get('TCHmin', 'NA')
            time_of_concentration_min = row.get('TCmin', 'NA')
            hydrologic_soil_group = row.get('HySGrp', 'NA')
            runoff_coefficient = row.get('C_Runoff', 'NA')
            rm_25yr_low = row.get('RM25yrL', 'NA')
            rm_25yr_expected = row.get('RM25yrE', 'NA')
            rm_25yr_upper = row.get('RM25yrU', 'NA')
            rm_50yr_low = row.get('RM50yrL', 'NA')
            rm_50yr_expected = row.get('RM50yrE', 'NA')
            rm_50yr_upper = row.get('RM50yrU', 'NA')
            rm_100yr_low = row.get('RM100yrL', 'NA')
            rm_100yr_expected = row.get('RM100yrE', 'NA')
            rm_100yr_upper = row.get('RM100yrU', 'NA')

            # Tooltip content for each polygon
            tooltip_content = f"""
            <b>Pour Point ID:</b> {point_id}<br>
            <b>Name of Drainage Structure:</b> {culvert_id}<br>
            <b>Gauged WS ID:</b> {WS_id}<br>
            <b>Shape of Drainage Structure:</b> {culvert_shape}<br>
            <b>Material of Drainage Structure:</b> {material}<br>
            <b>Width:</b> {width} ft<br>
            <b>Height:</b> {height} ft<br>
            <b>Gauging Station Flag:</b> {flag_gauging_station}<br>
            <b>Group ID:</b> {paired_group_ID}<br>
            <b>Group Size:</b> {paired_group_size}<br>
            <b>Average Slope:</b> {avg_sl_percentage}%<br>
            <b>Longest Channel Length:</b> {channel_length_m} m<br>
            <b>Max. Overland flow Length:</b> {overflow_length_m} m<br>
            <b>Time of Overland Flow:</b> {time_of_overflow_min} min<br>
            <b>Time of Main Channel Flow:</b> {time_of_channel_min} min<br>
            <b>Total Time of Concentration:</b> {time_of_concentration_min} min<br>
            <b>Hydrologic Soil Group:</b> {hydrologic_soil_group}<br>
            <b>Runoff Coefficient:</b> {runoff_coefficient}<br>
            <b>Rational Method 25-Year (m3/s):</b> CI-Low: {rm_25yr_low}<br> 
            <b>Rational Method 25-Year (m3/s):</b> Expected: {rm_25yr_expected}<br>
            <b>Rational Method 25-Year (m3/s):</b> CI-Upper: {rm_25yr_upper}<br>
            <b>Rational Method 50-Year (m3/s):</b> CI-Low: {rm_50yr_low}<br> 
            <b>Rational Method 50-Year (m3/s):</b> Expected: {rm_50yr_expected}<br>
            <b>Rational Method 50-Year (m3/s):</b> CI-Upper: {rm_50yr_upper}<br>
            <b>Rational Method 100-Year (m3/s):</b> CI-Low: {rm_100yr_low}<br> 
            <b>Rational Method 100-Year (m3/s):</b> Expected: {rm_100yr_expected}<br>
            <b>Rational Method 100-Year (m3/s):</b> CI-Upper: {rm_100yr_upper}<br>
            """


            # Add polygon to map with tooltip
            folium.GeoJson(
                row.geometry,
                name='Watershed Polygon',
                style_function=lambda feature: {
                    'fillColor': random_color(),
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.2,
                    'dashArray': '5, 5'
                },
                tooltip=tooltip_content
            ).add_to(watershed_group)

        # Add the new watershed group to the base map
        watershed_group.add_to(base_map)
    # Return the updated base map
    return base_map

# ================================================================================================================================================
# *************************************  Main functions ********************************************************
# ============================================================================================================

# ---------------------------------------------------------------------------------------------------
# **************************Case 1: PeakQ_using_PI_from_NOAA_runoff_coeff_from_table --- No Data Available *****************************************  
# ---------------------------------------------------------------------------------------------------
def PeakQ_using_PI_from_NOAA_runoff_coeff_from_table(save_ws_char_path,
                                                        boundary_polygon_path,
                                                        gssurgo_base_raster_path,
                                                        usa_states_shapefile_path,
                                                        dem_UTM_reprojected_path,
                                                        input_watersheds_path,
                                                        output_ws_path,
                                                        RM_output_folder,
                                                        aoi_soil_data_path,
                                                        runoff_table_path,
                                                        temp_folder_path,
                                                        crunof_output_folder,
                                                        clipped_nlcd_raster_path,
                                                        NLCD_CONUS_raster_path,
                                                        base_map,
                                                        rp_list,
                                                        pointPI,
                                                        name=None,
                                                        user_id=None,
                                                        project_name=None,
                                                        task_type=None,
                                                        check_cancellation_func=None):
    try:
        if not os.path.exists(save_ws_char_path):
            
            gssurgo_raster_to_clipped_polygon(boundary_polygon_path, gssurgo_base_raster_path, aoi_soil_data_path, 
                             usa_states_shapefile_path, state_abbr_column="stusps")
            
            assign_dominant_hydrologic_soil_group_to_watersheds_using_GSSURGO(aoi_soil_data_path,
                                                                            input_watersheds_path,
                                                                            output_ws_path)
            print('hydro soil group assigned to each watershed')
        else:
            output_ws_path = save_ws_char_path
        
            
            
        check_cancellation_func(user_id, project_name, task_type)
        calculate_runoff_coefficient_from_table(output_ws_path,dem_UTM_reprojected_path,
                                                runoff_table_path,aoi_soil_data_path,
                                                temp_folder_path ,crunof_output_folder,
                                                clipped_nlcd_raster_path,
                                                NLCD_CONUS_raster_path,
                                                boundary_polygon_path)
        print('runoff coefficient calculated')
        check_cancellation_func(user_id, project_name, task_type)
        gdf_peakQ=Calculate_peakQ_using_RM_from_NOAA_Atlas(crunof_output_folder,
                                                        RM_output_folder,
                                                        temp_folder_path,
                                                        pointPI,
                                                        rp_list,
                                                        ts_type='ams')
        print('Rational method Analysis Completed')
        check_cancellation_func(user_id, project_name, task_type)
        # folium_map = add_layers_to_basemap(base_map,gdf_peakQ,name)
        
        return gdf_peakQ
    except TaskCancelledError:
        # Re-raise to let the calling function handle it
        raise
    except Exception as e:
        print(f"Error in rational method: {str(e)}")

# ---------------------------------------------------------------------------------------------------
# **************************Case2:  PeakQ_using_PI_from_WS_runoff_coeff_from_table ***************************************** 
# --------------------------------------------------------------------------------------------------- 
def PeakQ_using_PI_from_WS_runoff_coeff_from_table(save_ws_char_path,
                                                   boundary_polygon_path,
                                                   gssurgo_base_raster_path,
                                                   usa_states_shapefile_path,
                                                   dem_UTM_reprojected_path,
                                                    input_watersheds_path,
                                                    input_pidf_output_dir_path,
                                                    output_ws_path,
                                                    aoi_soil_data_path,
                                                    runoff_table_path,
                                                    temp_folder_path,
                                                    output_folder,
                                                    clipped_nlcd_raster_path,
                                                    NLCD_CONUS_raster_path,
                                                    base_map,
                                                    pointPI,
                                                    name=None,
                                                    user_id=None,
                                                    project_name=None,
                                                    task_type=None,
                                                    check_cancellation_func=None):
    
    try:
        if not os.path.exists(save_ws_char_path):
            
            gssurgo_raster_to_clipped_polygon(boundary_polygon_path, gssurgo_base_raster_path, aoi_soil_data_path, 
                             usa_states_shapefile_path, state_abbr_column="stusps")
            
            assign_dominant_hydrologic_soil_group_to_watersheds_using_GSSURGO(aoi_soil_data_path,
                                                                            input_watersheds_path,
                                                                            output_ws_path)
            print('hydro soil group assigned to each watershed')
        else:
            output_ws_path = save_ws_char_path
            
            
        check_cancellation_func(user_id, project_name, task_type)
        calculate_runoff_coefficient_from_table(output_ws_path,
                                                dem_UTM_reprojected_path,
                                                runoff_table_path,
                                                aoi_soil_data_path,
                                                temp_folder_path,
                                                output_folder,
                                                clipped_nlcd_raster_path,
                                                NLCD_CONUS_raster_path,
                                                boundary_polygon_path)
        print('runoff coefficient calculated')
        check_cancellation_func(user_id, project_name, task_type)
        gdf_peakQ=Calculate_peakQ_using_RM_PI_from_WS(output_folder, 
                                            input_pidf_output_dir_path, 
                                            output_folder)
        print('Rational method Analysis Completed')
        check_cancellation_func(user_id, project_name, task_type)
        # folium_map = add_layers_to_basemap(base_map,gdf_peakQ,name)
        
        return gdf_peakQ
    except TaskCancelledError:
        # Re-raise to let the calling function handle it
        raise
    except Exception as e:
        print(f"Error in rational method: {str(e)}")    


# ---------------------------------------------------------------------------------------------------  
# ======================  Case3: PeakQ_using_PI_from_NOAA_runoff_coeff_back_calc ==========================================================================================================================
# ---------------------------------------------------------------------------------------------------
def PeakQ_using_PI_from_NOAA_runoff_coeff_back_calc(stream_data_dir_path,
                                                    precip_data_dir_path,
                                                    single_site_shapefile_path, 
                                                    ws_char_path, 
                                                    roi_df_dir_path,
                                                    RM_output_folder,
                                                    temp_folder_path,
                                                    crunof_output_folder,
                                                    base_map,
                                                    rp_list,
                                                    Gst_Names,
                                                    pointPI,
                                                    name=None,
                                                    select_top=10000,
                                                    user_id=None,
                                                    project_name=None,
                                                    task_type=None,
                                                    check_cancellation_func=None):
    try:
        cval_df = back_calculate_runoff_for_gauged_WSs(stream_data_dir_path, 
                                            precip_data_dir_path,
                                            single_site_shapefile_path,
                                            ws_char_path, 
                                            roi_df_dir_path,
                                            crunof_output_folder, 
                                            Gst_Names, select_top)
        check_cancellation_func(user_id, project_name, task_type)
        gdf_peakQ=Calculate_peakQ_using_RM_from_NOAA_Atlas(crunof_output_folder,
                                                        RM_output_folder,
                                                        temp_folder_path,
                                                        pointPI,
                                                        rp_list,
                                                        ts_type='ams')
        print('Rational method Analysis Completed')
        check_cancellation_func(user_id, project_name, task_type)
        # folium_map = add_layers_to_basemap(base_map,gdf_peakQ,name)
            
        return gdf_peakQ
    except TaskCancelledError:
        # Re-raise to let the calling function handle it
        raise
    except Exception as e:
        print(f"Error in rational method: {str(e)}")


# ---------------------------------------------------------------------------------------------------  
# ====================== Case4:  PeakQ_using_PI_from_WS_runoff_coeff_back_calc  ==========================================================================================================================
# ---------------------------------------------------------------------------------------------------
def PeakQ_using_PI_from_WS_runoff_coeff_back_calc(stream_data_dir_path,
                                                    precip_data_dir_path,
                                                    single_site_shapefile_path, 
                                                    ws_char_path, 
                                                    roi_df_dir_path,
                                                    RM_output_folder,
                                                    crunof_output_folder,
                                                    input_pidf_output_dir_path,
                                                    base_map,
                                                    Gst_Names,
                                                    pointPI,
                                                    name=None,
                                                    select_top=10000,
                                                    user_id=None,
                                                    project_name=None,
                                                    task_type=None,
                                                    check_cancellation_func=None):
    try:
        cval_df = back_calculate_runoff_for_gauged_WSs(stream_data_dir_path, 
                                            precip_data_dir_path,
                                            single_site_shapefile_path,
                                            ws_char_path, 
                                            roi_df_dir_path,
                                            crunof_output_folder, 
                                            Gst_Names, 
                                            select_top)
        check_cancellation_func(user_id, project_name, task_type)
        print('runoff coefficient calculated')
        gdf_peakQ=Calculate_peakQ_using_RM_PI_from_WS(crunof_output_folder, 
                                            input_pidf_output_dir_path, 
                                            RM_output_folder)
        print('Rational method Analysis Completed')
        check_cancellation_func(user_id, project_name, task_type)
        # folium_map = add_layers_to_basemap(base_map,gdf_peakQ,name)
            
        return gdf_peakQ
    except TaskCancelledError:
        # Re-raise to let the calling function handle it
        raise
    except Exception as e:
        print(f"Error in rational method: {str(e)}")
