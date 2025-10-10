########### Data structure and dataframe
import numpy as np
import pandas as pd

########### Geospatial analysis
import geopandas as gpd
from shapely.geometry import Point, box, MultiPolygon, shape
from osgeo import gdal, ogr, osr
import rasterio
from rasterio import plot
from rasterio.features import rasterize, shapes
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling

########### File download and handling OS paths
import requests
import zipfile
import sys
import os
import shutil
from pathlib import Path

########## watershed analysis
import whitebox

wbt = whitebox.WhiteboxTools()

########## Plotting and visualization
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import pyplot as plt
from app import TaskCancelledError

# A decent literature on that: discusses various morphometric characteristics of watershed.
# Link to the paper: https://www.sciencedirect.com/science/article/pii/S258947142300030X#s0010.
# Step 1:
# ---------------------------------------Function to download data from url and filename

# # Example usage:
# url = "https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/SC_geopackage_wetlands.zip"
# file_path = "/content/SC_geopackage_wetlands.zip"  # Specify desired location
# download_data(url, file_path)

def download_data(url, file_path):
  """Downloads a data file from a given URL to a specified file path.

  Args:
      url (str): The URL of the data file.
      file_path (str): The desired file path for the downloaded file, including filename.

  Raises:
      requests.exceptions.RequestException: If there's an error during the download.
  """
  try:
      response = requests.get(url)
      response.raise_for_status()
      
      with open(file_path, 'wb') as f:
          f.write(response.content)

    #   print(f"Successfully downloaded {filename} to {file_path}")

  except requests.exceptions.RequestException as e:
    #   print(f"Error downloading {filename}: {e}")
      raise
  
# Step 2:
# --------------------------------------- Function to unzip and save a folder using path
def unzip_file(file_path, destination_folder):
    """
    Unzips a zip file to a specified destination folder.

    Args:
        file_path (str): Path to the zip file.
        destination_folder (str): Path to the folder where the extracted files will be placed.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
            print(f"Successfully unzipped {file_path} to {destination_folder}")
    except Exception as e:
        print(f"Error unzipping {file_path}: {e}")


# # Example usage
# file_path = '/content/gSSURGO_SC.zip'
# destination_folder = '/content/gSSURGO_SC'
# unzip_file(file_path, destination_folder)
# Step 3
# ----------------------------------------- Function to resample raster data ----------------
def resample_raster(
    input_raster_data,
    input_raster_transform,
    input_raster_crs,
    target_raster_array,
    target_raster,
):
    """Resample the input_raster_data to match the target_array and target_raster transform using nearest neighbor resampling.
    This function:
    1. Reads Input Raster Data: Reads input raster data, transform, and CRS.
    2. Initializes Resampled Raster: Creates an empty resampled raster array.
    3. Reprojects Raster Data: Resamples input raster data onto target raster using nearest neighbor resampling.
    4. Returns Resampled Raster: Returns the resampled raster data.

    Parameters:
    - input_raster_data (numpy array): Input raster data.
    - input_raster_transform (affine transform): Input raster's affine transformation.
    - input_raster_crs (CRS): Input raster's Coordinate Reference System.
    - target_raster_array (numpy array): Target raster array.
    - target_raster (rasterio RasterReader): Target raster.

    Returns:
    - resampled_raster (numpy array): Resampled raster data.

    Libraries Used:
    - numpy
    - rasterio
    - rasterio.warp
    """

    resampled_raster = np.empty_like(target_raster_array)
    reproject(
        source=input_raster_data,
        destination=resampled_raster,
        src_transform=input_raster_transform,
        src_crs=input_raster_crs,
        dst_transform=target_raster.transform,
        dst_crs=target_raster.crs,
        resampling=Resampling.nearest,
    )
    return resampled_raster


# ----------------------------------------------Function to clip vector to polygon ----------------------
def clip_polygon_to_boundary(vector_data_gdf, boundary_polygon_file_path):
    """
    ClipsPolygon vector data to a specified boundary polygon.

    This function:
    1. Loads Boundary Polygon: Reads boundary polygon from file.
    2. Transforms Vector Data: Converts vector data to boundary polygon's CRS.
    3. Clips Vector Data: Clips vector data to boundary polygon extent.
    4. Returns Clipped Data: Returns clipped vector data.

    Parameters:
    - vector_data_gdf (geopandas GeoDataFrame): Input Polygon vector data.
    - boundary_polygon_file_path (str): Path to boundary polygon file.

    Returns:
    - clipped_gdf (geopandas GeoDataFrame): Clipped vector data.

    Libraries Used:
    - geopandas: For reading, manipulating and performing spatial operations on geospatial vector data.
    """

    # Load the boundary shapefile
    boundary_gdf = gpd.read_file(boundary_polygon_file_path)
    boundary_crs = boundary_gdf.crs
    vector_data_gdf = vector_data_gdf.to_crs(boundary_crs)
    # Clip vector data
    clipped_gdf = gpd.clip(vector_data_gdf, boundary_gdf)

    return clipped_gdf


def merging_watersheds_and_pour_point_characteristics(
    filtered_watershed_delin_polygon_path,
    filtered_pour_delin_point_path,
    save_watershed_polygon_path,
    pour_ID="Point_ID",
):
    import geopandas as gpd
    import pandas as pd

    # Step 1: Read watershed polygons
    watersheds_gdf = gpd.read_file(filtered_watershed_delin_polygon_path)

    # Step 2: Read selected pour points
    selected_culverts_gdf = gpd.read_file(filtered_pour_delin_point_path).to_crs("EPSG:4326")

    # Step 3: Merge watershed and pour geometry
    watershed_gdf_merged = watersheds_gdf.copy()
    watershed_gdf_merged = pd.merge(
        watershed_gdf_merged,
        selected_culverts_gdf[[pour_ID, "geometry"]],
        on=pour_ID,
        how="left",
        suffixes=("_x", "_y"),
    )

    # Step 4: Extract coordinates and fix geometry
    watershed_gdf_merged["Pour_lon"] = watershed_gdf_merged["geometry_y"].apply(
        lambda p: p.x if p is not None else None
    )
    watershed_gdf_merged["Pour_lat"] = watershed_gdf_merged["geometry_y"].apply(
        lambda p: p.y if p is not None else None
    )
    watershed_gdf_merged = watershed_gdf_merged.drop(columns=["geometry_y"])
    watershed_gdf_merged = watershed_gdf_merged.rename(columns={"geometry_x": "geometry"})
    watershed_gdf_merged = gpd.GeoDataFrame(
        watershed_gdf_merged, geometry="geometry", crs=watersheds_gdf.crs
    )

    # Step 5: Add GWS_ID column by default
    watershed_gdf_merged["GWS_ID"] = "NA"

    # Step 6: If Flag_Gst is present, assign GWS_ID based on matched pour_IDs
    if "Flag_Gst" in selected_culverts_gdf.columns:
        list_gauged_st_pour_ID = selected_culverts_gdf.loc[
            selected_culverts_gdf["Flag_Gst"] == 1, pour_ID
        ]
        list_gauged_pt_names = selected_culverts_gdf.loc[
            selected_culverts_gdf["Flag_Gst"] == 1, "GWS_ID"
        ]

        print("Gauged Pour IDs:", list_gauged_st_pour_ID.tolist())
        print("Gauged Point Names:", list_gauged_pt_names.tolist())

        for pour_id, station_name in zip(list_gauged_st_pour_ID, list_gauged_pt_names):
            watershed_gdf_merged.loc[
                watershed_gdf_merged[pour_ID].astype(str) == str(pour_id), "GWS_ID"
            ] = station_name

        print(f"Status: Added gauging stations to the watershed shapefile based on {pour_ID} column")
    else:
        print("Note: 'Flag_Gst' column not found in pour point data. Skipping gauging station assignment.")

    # Step 7: Save the merged watershed shapefile
    watershed_gdf_merged.to_file(
        save_watershed_polygon_path, driver="ESRI Shapefile", encoding="utf-8", mode="w"
    )
    print(f"Final watershed shapefile saved with columns: {list(watershed_gdf_merged.columns)}")


# merging_watersheds_and_pour_point_characteristics(    filtered_watershed_delin_polygon_path='/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
#                                                       filtered_pour_delin_point_path='/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_pour_point_by_area_UTM_reprojected.shp',
#                                                       save_watershed_polygon_path='/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp',
#                                                       save_pour_point_path='/content/drive/MyDrive/WS_Properties/filtered_pour_point_by_area_UTM_reprojected.shp',
#                                                       list_gauged_st_pour_ID=[253,573,792,791],
#                                                       list_gauged_pt_names=['WS77','WS78','WS79','WS80'],
#                                                       pour_ID='Point_ID',
#                                                       list_column_names_to_merge=['Culvert_ID','Purpose','CulvertSha', 'Material', 'Width_ft', 'Height_ft']
#                                                       )

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
        

# ----------------------------- Function to process and add mean elevation, slope%, and TWI to watershed shapefile
def calculate_watershed_mean_elev_slope_TWI(
    dem_path, flow_accumulation_path, watersheds_path, output_path, WS_ID
):
    """
    Calculates mean elevation, slope, and Topographic Wetness Index (TWI) for each watershed.

    This function:
    1. Loads Watershed Polygons: Reads watershed vector data.
    2. Loads DEM and Flow Accumulation Rasters: Opens DEM and flow accumulation raster files.
    3. Clips and Resamples Rasters: Clips and resamples DEM and flow accumulation rasters for each watershed.
    4. Calculates Statistics: Computes mean elevation, slope, and TWI for each watershed.
    5. Saves Output: Writes results to a GeoDataFrame.

    Parameters:
    - dem_path (str): Path to DEM raster file.
    - flow_accumulation_path (str): Path to flow accumulation raster file.
    - watersheds_path (str): Path to watershed vector file.
    - output_path (str): Path to save output GeoDataFrame.
    - WS_ID (str): Watershed ID column name.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing calculated statistics for each watershed.

    Libraries Used:
    - geopandas
    - rasterio
    - numpy
    """

    # Load the watershed polygons
    watersheds = gpd.read_file(watersheds_path)

    # Open the DEM raster and the flow accumulation raster
    with rasterio.open(dem_path) as dem, rasterio.open(
        flow_accumulation_path
    ) as flow_accumulation_raster:
        # Ensure CRS match
        if dem.crs != watersheds.crs:
            watersheds = watersheds.to_crs(dem.crs)
        # Loop over each watershed polygon
        for index, (_, watershed) in enumerate(watersheds.iterrows()):
            # Clip flow accumulation raster to the current watershed
            watershed_geometry = [watershed.geometry]
            flow_accumulation_clipped, _ = mask(
                flow_accumulation_raster, watershed_geometry, crop=True
            )
            flow_accumulation_clipped = flow_accumulation_clipped[
                0
            ]  # Single band flow accumulation
            flow_accumulation_clipped = np.where(
                flow_accumulation_clipped == flow_accumulation_raster.nodata,
                np.nan,
                flow_accumulation_clipped,
            )

            # Clip DEM and resample to match the flow accumulation raster
            dem_clipped, _ = mask(dem, watershed_geometry, crop=True)
            dem_clipped = dem_clipped[0]  # Single band DEM
            dem_resampled = resample_raster(
                dem_clipped,
                dem.transform,
                dem.crs,
                flow_accumulation_clipped,
                flow_accumulation_raster,
            )

            # Remove no-data values
            dem_resampled = np.where(dem_resampled == dem.nodata, np.nan, dem_resampled)

            # Calculate mean elevation
            mean_elevation = np.nanmean(dem_resampled)

            # Calculate slope (in percentage) using central differences in x and y directions
            x, y = np.gradient(
                dem_resampled, dem.res[0], dem.res[1]
            )  # Gradient in x and y directions
            slope_radians = np.arctan(np.sqrt(x**2 + y**2))

            # Calculate Topographic Wetness Index (TWI)
            tan_slope = np.tan(slope_radians)
            twi = np.log(
                (flow_accumulation_clipped + 1e-6) / (tan_slope + 1e-6)
            )  # Avoid division by zero
            mean_twi = np.nanmean(twi)

            watersheds.loc[index, "AvgELm"] = mean_elevation
            watersheds.loc[index, "AvgTWI"] = mean_twi

    # Save the output to a shapefile
    watersheds.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8", mode="w")
    return watersheds


# Example Usage:
# input_watershed_path = '/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp'
# # gpd.read_file(input_watershed_path).head()
# output_path = '/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp'
# dem_path = '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/DEM_UTM_reprojected.tif'
# flow_accum_path = '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif'
# ws_stats_gdf = calculate_watershed_mean_elev_slope_TWI(dem_path,
#                                                        flow_accum_path,
#                                                        input_watershed_path,
#                                                        output_path,
#                                                        'Point_ID')
# ws_stats_gdf.head()


# ---------------------------------- Function to derive the watershed mean of PRISM : 30-yr precipitation normals
def zonal_average_of_raster(raster_path, dem_path, polygon_path, output_path, IDname):
    """
    Calculates zonal mean of pixel values for each polygon.
    This function:
    1. Loads Raster Data: Opens raster file.
    2. Loads DEM Data: Opens DEM raster to get target CRS and resolution.
    3. Reprojects and Resamples: Aligns input raster with DEM.
    4. Loads Polygon Data: Reads polygon vector data.
    5. Clips Raster to Polygons: Clips raster to each polygon.
    6. Calculates Zonal Mean: Computes mean pixel value for each polygon.
    7. Saves Output: Writes results to a GeoDataFrame.
    
    Parameters:
    - raster_path (str): Path to raster file.
    - dem_path (str): Path to DEM raster file for reprojection reference.
    - polygon_path (str): Path to polygon vector file.
    - output_path (str): Path to save output GeoDataFrame.
    - IDname (str): ID column name.
    
    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing calculated mean for each polygon.
    
    Libraries Used:
    - geopandas
    - rasterio
    - numpy
    - rasterio.warp
    """
    # Open the DEM raster to get target CRS and resolution
    with rasterio.open(dem_path) as dem:
        target_crs = dem.crs
        target_transform = dem.transform
        target_height = dem.height
        target_width = dem.width
        
    # Open the input raster
    with rasterio.open(raster_path) as src_raster:
        # Create a temporary in-memory raster with reprojected and resampled data
        resampled_data = np.zeros((target_height, target_width), dtype=src_raster.dtypes[0])
        
        # Reproject and resample using bilinear interpolation
        from rasterio.warp import reproject, Resampling
        reproject(
            source=src_raster.read(1),
            destination=resampled_data,
            src_transform=src_raster.transform,
            src_crs=src_raster.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        
        # Create a memory file to hold the reprojected raster
        from rasterio.io import MemoryFile
        memfile = MemoryFile()
        with memfile.open(
            driver='GTiff',
            height=target_height,
            width=target_width,
            count=1,
            dtype=src_raster.dtypes[0],
            crs=target_crs,
            transform=target_transform,
            nodata=src_raster.nodata
        ) as dst:
            dst.write(resampled_data, 1)
        
        # Read polygons
        zones = gpd.read_file(polygon_path)
        
        # Process each polygon
        for index, (_, zone) in enumerate(zones.iterrows()):
            # Get the geometry of the current zone
            zone_geometry = [zone.geometry]
            
            # Use the reprojected and resampled raster for masking
            with memfile.open() as prism:
                # Clip the raster to the current polygon
                prism_clipped, transform = mask(prism, zone_geometry, crop=True)
                
                # Remove no-data values
                prism_clipped = np.where(
                    prism_clipped == prism.nodata, np.nan, prism_clipped
                )
                
                # Calculate mean value
                mean_value = np.nanmean(prism_clipped)
                
                # Store the result in the GeoDataFrame
                zones.loc[index, "N30PRin"] = mean_value
        
        # Save the output
        zones.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8", mode="w")
        
        return zones


# Example Usage:
# raster_path = '/content/drive/MyDrive/WS_Properties/SANTEE_30yr_precip_Normal_1m_UTM.tif'
# output_path = '/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp'
# polygon_path = '/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp'
# ppt_30yr_prism_gdf = zonal_average_of_raster(raster_path,
#                                              polygon_path,
#                                              output_path,
#                                              'Point_ID')
# ppt_30yr_prism_gdf.head()
# =======================================================================================================================
# clip and merge the gssurgo dominant hydrologic group variable to use it in determining the CN values for each watershed
# =========================================================================================================================
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
# =======================================================================================================================================================================================================================
# -----------------------------------Function to download and process Hydro Soil Group from GSSURGO data. This function also assigns the most dominant (covering most area) soil group for the sub-watersheds.
# ======================================================================================================================================================================================================================================
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
#                                                                   aoi_gSSURGO_hydgrpdcd_data_path='/content/drive/MyDrive/WS_Properties/gSSURGO_data_polygon_UTM_reprojected.shp'
#                                                                   )


# -----------------------------------Function to calculate Curve number from Hydrologic soil group and NLCD land cover data
# Reference Codes on how to download NLCD using python https://github.com/reirby/LandCoverDownloader/blob/main/README.md
# Curve number was assigned based on : https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/gis-tutorials-and-guides/creating-a-curve-number-grid-and-computing-subbasin-average-curve-number-values
# -----------------------------------Function to assign dominant CN value to each watershed

def create_cn_raster(
    boundary_shapefile_path,
    gssurgo_polygon_shapefile_path,
    NLCD_2024_CONUS_raster_path,
    cn_table_csv_path,
    temp_folder_path,
    clipped_nlcd_raster_path,
    save_reprojected_CN_raster_path,
):
    try:
        # Step 1: Get bounds
        gdf_boundary = gpd.read_file(boundary_shapefile_path)
        boundary_orig_crs = gdf_boundary.crs
        
        with rasterio.Env(GDAL_CACHEMAX=64):
            with rasterio.open(NLCD_2024_CONUS_raster_path) as src:
                # Reproject AOI to raster CRS
                gdf_boundary_clip = gdf_boundary.to_crs(src.crs)
                shapes = [geom.__geo_interface__ for geom in gdf_boundary_clip.geometry]
                
                # Clip (only reads required blocks)
                data, transform = rasterio.mask.mask(src, shapes, crop=True)
                
                # Determine appropriate nodata value based on data type
                dtype = src.dtypes[0]
                if dtype in ['uint8', 'uint16', 'uint32']:
                    # For unsigned integer types, use 0 or max value as nodata
                    if dtype == 'uint8':
                        nodata_value = 255  # Use max value for uint8
                    elif dtype == 'uint16':
                        nodata_value = 65535
                    else:  # uint32
                        nodata_value = 4294967295
                elif dtype in ['int8', 'int16', 'int32']:
                    # For signed integer types, can use negative values
                    nodata_value = -9999 if dtype != 'int8' else -128
                else:
                    # For float types
                    nodata_value = -9999.0
                
                # Create clean profile
                profile = {
                    'driver': 'GTiff',
                    'dtype': dtype,
                    'nodata': nodata_value,
                    'width': data.shape[2],
                    'height': data.shape[1],
                    'count': 1,
                    'crs': src.crs,
                    'transform': transform
                }
        
        # Save clipped raster
        with rasterio.open(clipped_nlcd_raster_path, "w", **profile) as dst:
            dst.write(data)
        
        # Step 2: Load the raster and polygon
        with rasterio.open(clipped_nlcd_raster_path) as src:
            raster_data = src.read(1)
            raster_transform = src.transform
            raster_crs = src.crs
            raster_resolution = src.res[0]

        gdf = gpd.read_file(gssurgo_polygon_shapefile_path)
        gdf = gdf.to_crs(raster_crs)
        value_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "A/D": 4, "B/D": 4, "C/D": 4}
        gdf["mapped_value"] = gdf["hydgrpdcd"].map(value_mapping)

        # Use a different nodata value for polygon raster (int32 can handle -9999)
        polygon_nodata_value = -9999
        polygon_raster = rasterize(
            [(geometry, attr) for geometry, attr in zip(gdf.geometry, gdf["mapped_value"])],
            out_shape=raster_data.shape,
            transform=raster_transform,
            fill=polygon_nodata_value,
            all_touched=True,
            dtype="int32",
        )

        # Step 3: Load the CSV file containing CN values
        # Obtained from https://github.com/WikiWatershed/tr-55/blob/develop/tr55/tables.py
        df = pd.read_csv(cn_table_csv_path)
        # Example structure of DataFrame `df`
        # Land Use Description | Land Use Value | A  | B  | C  | D
        # ---------------------------------------------------------
        # Urban               | 11             | 70 | 75 | 80 | 85
        # Agriculture         | 12             | 68 | 74 | 79 | 83
        # Forest              | 13             | 55 | 60 | 65 | 70
        # Convert the DataFrame into a lookup dictionary for fast access
        cn_lookup = {(row["Land Use Value"], 1): row["A"] for _, row in df.iterrows()}
        cn_lookup.update({(row["Land Use Value"], 2): row["B"] for _, row in df.iterrows()})
        cn_lookup.update({(row["Land Use Value"], 3): row["C"] for _, row in df.iterrows()})
        cn_lookup.update({(row["Land Use Value"], 4): row["D"] for _, row in df.iterrows()})

        # Step 4: Initialize the new raster to store CN values (int32 can handle -9999)
        cn_nodata_value = -9999
        cn_raster = np.full(raster_data.shape, cn_nodata_value, dtype="int32")

        # Step 5: Assign CN values based on raster_values and polygon_values
        # Handle the case where original raster might have its own nodata value
        original_nodata = profile['nodata']
        
        # Create mask for valid data (exclude both original nodata and polygon nodata)
        valid_mask = (polygon_raster != polygon_nodata_value)
        if original_nodata is not None:
            valid_mask = valid_mask & (raster_data != original_nodata)
        
        for (r_value, p_value), (row, col) in zip(
            zip(raster_data[valid_mask], polygon_raster[valid_mask]), 
            np.argwhere(valid_mask)
        ):
            cn_value = cn_lookup.get((int(r_value), int(p_value)), cn_nodata_value)
            cn_raster[row, col] = cn_value

        temp_save_CN_raster_path = os.path.join(temp_folder_path, 'CN_raster.tif')
        
        # Step 6: Save the new raster with CN values
        with rasterio.open(
            temp_save_CN_raster_path,
            "w",
            driver="GTiff",
            height=cn_raster.shape[0],
            width=cn_raster.shape[1],
            count=1,
            dtype=cn_raster.dtype,
            crs=raster_crs,
            transform=raster_transform,
            nodata=cn_nodata_value,
        ) as dst:
            dst.write(cn_raster, 1)

        # Step 7: Reproject CN raster to boundary CRS
        with rasterio.open(temp_save_CN_raster_path) as src:
            raster_data = src.read(1)
            raster_transform = src.transform
            raster_crs = src.crs

        # Reproject the raster to the same coordinate system as the polygon
        if raster_crs != gdf_boundary.crs:
            # Calculate the transform for the new raster
            new_transform, new_width, new_height = (
                rasterio.warp.calculate_default_transform(
                    raster_crs,
                    gdf_boundary.crs,
                    raster_data.shape[1],
                    raster_data.shape[0],
                    *src.bounds,
                )
            )

            reprojected_raster = np.empty(
                shape=(new_height, new_width), dtype=raster_data.dtype
            )

            # Reproject the raster data
            reproject(
                source=raster_data,
                destination=reprojected_raster,
                src_transform=raster_transform,
                src_crs=raster_crs,
                dst_transform=new_transform,
                dst_crs=gdf_boundary.crs,
                resampling=Resampling.nearest,
                src_nodata=cn_nodata_value,
                dst_nodata=cn_nodata_value,
            )
        else:
            reprojected_raster = raster_data
            new_transform = raster_transform
            new_width, new_height = raster_data.shape[1], raster_data.shape[0]

        # Save the reprojected raster to a new file
        with rasterio.open(
            save_reprojected_CN_raster_path,
            "w",
            driver="GTiff",
            height=reprojected_raster.shape[0],
            width=reprojected_raster.shape[1],
            count=1,
            dtype=reprojected_raster.dtype,
            crs=gdf_boundary.crs,
            transform=new_transform,
            nodata=cn_nodata_value,
        ) as dst:
            dst.write(reprojected_raster, 1)
        
        # Step 8: Reproject the clipped NLCD back to original boundary CRS
        with rasterio.open(clipped_nlcd_raster_path) as src:
            if boundary_orig_crs is not None and src.crs != boundary_orig_crs:
                dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                    src.crs, boundary_orig_crs, src.width, src.height, *src.bounds
                )
                
                # Determine appropriate nodata for reprojection based on original dtype
                src_dtype = src.dtypes[0]
                if src_dtype in ['uint8', 'uint16', 'uint32']:
                    if src_dtype == 'uint8':
                        reproject_nodata = 255
                    elif src_dtype == 'uint16':
                        reproject_nodata = 65535
                    else:  # uint32
                        reproject_nodata = 4294967295
                else:
                    reproject_nodata = src.nodata if src.nodata is not None else -9999
                
                kwargs = {
                    'driver': 'GTiff',
                    'dtype': src_dtype,
                    'nodata': reproject_nodata,
                    'width': dst_width,
                    'height': dst_height,
                    'count': 1,
                    'crs': boundary_orig_crs,
                    'transform': dst_transform
                }

                tmp_out = clipped_nlcd_raster_path + ".tmp.tif"
                with rasterio.open(tmp_out, "w", **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=boundary_orig_crs,
                        resampling=Resampling.nearest,
                        src_nodata=src.nodata,
                        dst_nodata=reproject_nodata,
                    )

                # Atomically replace original clipped_nlcd with the reprojected one
                if os.path.exists(tmp_out):
                    os.replace(tmp_out, clipped_nlcd_raster_path)

        return None

    except Exception as e:
        print(f"Unexpected error in CN raster creation: {e}", file=sys.stderr)
        raise
# -----------------------------------Function to calculate the wetland cover area for each watershed
def find_weighted_avg_area_cn_value(cn_raster_path, shapefile_path):
    """
    Finds maximum area Curve Number (CN) value for each polygon.

    This function:
    1. Loads shapefile and CN raster data.
    2. Iterates over each polygon.
    3. Masks CN raster data to polygon extent.
    4. Finds CN value covering maximum area.
    5. Assigns max area CN value to shapefile.
    6. Saves updated shapefile.

    Note: Assumes -9999 as raster no-data value.

    Parameters:
    - cn_raster_path (str): Path to CN raster file.
    - shapefile_path (str): Path to shapefile.

    Returns:
    - geopandas.GeoDataFrame: Updated shapefile with max area CN values.

    Libraries Used:
    - geopandas
    - rasterio
    - numpy
    """

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Load the raster data
    with rasterio.open(cn_raster_path) as src:
        no_data_value = -9999  # Assuming this is the no-data value in your raster

        # Iterate over each polygon in the GeoDataFrame
        for index, polygon in gdf.iterrows():
            # Create a mask for the current polygon
            masked_raster, _ = rasterio.mask.mask(
                src, [polygon.geometry], crop=True, nodata=no_data_value
            )

            # Flatten the masked raster and remove no-data values
            values, counts = np.unique(
                masked_raster[masked_raster != no_data_value], return_counts=True
            )

            if values.size > 0:  # Check if there are any valid values
                # Filter out NaN values explicitly to ensure np.mean works correctly
                valid_values = values[~np.isnan(values)]
                
                if valid_values.size > 0:  # Check if there are still valid values after NaN removal
                    # Calculate the mean CN value and round up to the next integer
                    mean_cn = np.mean(valid_values)
                    rounded_mean_cn = int(np.ceil(mean_cn))  # Ceiling rounding (e.g., 40.5 -> 41)
                    # Assign the rounded-up mean to the GeoDataFrame
                    gdf.loc[index, "CN_val"] = rounded_mean_cn
                    # Optional: Print for debugging
                    # print(f"Polygon {index}: Mean CN value = {mean_cn}, Rounded up = {rounded_mean_cn}")
                else:
                    # Handle case where all values are NaN after filtering
                    gdf.loc[index, "CN_val"] = np.nan
            else:
                print(f"Polygon {index}: No valid CN values found.")
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="utf-8", mode="w")
    return gdf


# ----------------------------------------Function to calculate the wetland cover area for each watershed
# Wetland cover data downloaded from
# https://www.fws.gov/program/national-wetlands-inventory/download-state-wetlands-data  
def wetland_cover_area(
    polygon_shapefile_path,
    base_wetland_data_path,
    usa_states_shapefile_path,
    save_watershed_polygon_path
):
    """
    Calculates wetland cover area for a watershed AOI by merging wetland data
    for all US states that the AOI crosses.

    Args:
        polygon_shapefile_path (str): Path to the zipped shapefile containing the boundary polygon.
        usa_states_shapefile_path (str): Path to a known US states shapefile (for state abbreviations).
        save_watershed_polygon_path (str): Path to save the watershed polygon shapefile.
        

    Returns:
        GeoDataFrame: Watershed GeoDataFrame with a new column 'WetAHa' (wetland area in hectares).
    """
    watersheds_gdf = gpd.read_file(polygon_shapefile_path)
    
    # Initialize WetAHa column with zeros at the start
    watersheds_gdf['WetAHa'] = 0.0
    
    # Get the list of states that intersect with the boundary polygon
    state_abbr_list = get_us_states_crossed(polygon_shapefile_path, usa_states_shapefile_path, state_abbr_column="stusps")
    
    if not state_abbr_list:
        print("No states found that intersect with the boundary polygon.")
        print("Setting WetAHa to 0 for all watersheds.")
        watersheds_gdf.to_file(save_watershed_polygon_path, driver='ESRI Shapefile', encoding='utf-8', mode='w')
        return watersheds_gdf
    
    print(f"Found {len(state_abbr_list)} states: {state_abbr_list}")
    
    # Initialize as empty list to collect individual GeoDataFrames
    wetlands_gdf_list = []
    ws_crs = watersheds_gdf.crs
    
    for state_abbr in state_abbr_list:
        # Construct the wetland shapefile path for this state
        wetland_aoi_path = os.path.join(base_wetland_data_path, state_abbr, 'wetland_polygon.zip')
        
        try:
            # Read clipping shapefile
            clip_ws_gdf = watersheds_gdf.to_crs('EPSG:5070')  # NAD83 / Conus Albers
            
            # Fix any invalid geometries in the watershed data
            clip_ws_gdf['geometry'] = clip_ws_gdf['geometry'].buffer(0)
            clip_poly = clip_ws_gdf.unary_union
            
            minx, miny, maxx, maxy = clip_poly.bounds

            # Read only features in bounding box
            gdf = gpd.read_file(wetland_aoi_path, bbox=(minx, miny, maxx, maxy))
            
            # Fix any invalid geometries in wetland data
            gdf['geometry'] = gdf['geometry'].buffer(0)
            print('Wetland data successfully read')

            # Then clip precisely with error handling
            try:
                wetlands_gdf_individual = gpd.clip(gdf, clip_poly).to_crs(ws_crs)
            except Exception as clip_error:
                print(f"Clipping error for {state_abbr}: {clip_error}")
                # Try alternative clipping method using overlay
                try:
                    clip_ws_gdf_temp = clip_ws_gdf.copy()
                    clip_ws_gdf_temp['clip_id'] = 1
                    gdf['wet_id'] = range(len(gdf))
                    wetlands_gdf_individual = gpd.overlay(gdf, clip_ws_gdf_temp, how='intersection').to_crs(ws_crs)
                    print(f"Successfully clipped using overlay method for {state_abbr}")
                except Exception as overlay_error:
                    print(f"Overlay method also failed for {state_abbr}: {overlay_error}")
                    continue
            
            # Optional: Add state identifier column
            wetlands_gdf_individual['state'] = state_abbr
            
            # Append to list
            wetlands_gdf_list.append(wetlands_gdf_individual)
            
            print(f"Loaded {len(wetlands_gdf_individual)} features from {state_abbr}")
            
        except Exception as e:
            print(f"Error loading wetlands for {state_abbr}: {e}")
            continue

    # Merge all GeoDataFrames into one
    if wetlands_gdf_list:
        wetlands_gdf_aoi = pd.concat(wetlands_gdf_list, ignore_index=True)
        print(f"Successfully merged {len(wetlands_gdf_aoi)} total wetland features from {len(wetlands_gdf_list)} states")
    else:
        print("No wetland data loaded. Setting WetAHa to 0 for all watersheds.")
        watersheds_gdf.to_file(save_watershed_polygon_path, driver='ESRI Shapefile', encoding='utf-8', mode='w')
        return watersheds_gdf
    
    # Iterate over each polygon in the shapefile and clip the GeoDataFrame
    for index, row in watersheds_gdf.iterrows():
        try:
            clip_polygon = row['geometry']
            
            # Validate and fix the clip polygon geometry if needed
            if not clip_polygon.is_valid:
                print(f"Warning: Invalid geometry found for watershed {index}. Attempting to fix...")
                clip_polygon = clip_polygon.buffer(0)
            
            # Perform clipping with error handling
            try:
                gdf_clipped = gpd.clip(wetlands_gdf_aoi, clip_polygon)
            except Exception as clip_error:
                print(f"Clipping error for watershed {index}: {clip_error}. Trying alternative method...")
                # Alternative: use spatial intersection
                try:
                    gdf_clipped = wetlands_gdf_aoi[wetlands_gdf_aoi.intersects(clip_polygon)].copy()
                    gdf_clipped['geometry'] = gdf_clipped['geometry'].intersection(clip_polygon)
                except Exception as intersect_error:
                    print(f"Intersection method also failed for watershed {index}: {intersect_error}")
                    watersheds_gdf.loc[index, 'WetAHa'] = 0
                    continue
            
            if gdf_clipped.empty:
                # Calculate the area of wetland cover within each watershed polygon
                watersheds_gdf.loc[index, 'WetAHa'] = 0  ## no wetland cover
            else:
                # Calculate the area of wetland cover within each watershed polygon
                # Filter out any invalid geometries before calculating area
                valid_geoms = gdf_clipped[gdf_clipped.geometry.is_valid]
                if len(valid_geoms) > 0:
                    watersheds_gdf.loc[index, 'WetAHa'] = (valid_geoms.area * 0.0001).sum()  ## *0.0001 converts area in m2 to Ha
                else:
                    watersheds_gdf.loc[index, 'WetAHa'] = 0
                    
        except Exception as e:
            print(f"Error processing watershed {index}: {e}")
            watersheds_gdf.loc[index, 'WetAHa'] = 0
            continue

    watersheds_gdf.to_file(save_watershed_polygon_path, driver='ESRI Shapefile', encoding='utf-8', mode='w')
    print(f"Watershed shapefile with WetAHa column saved to {save_watershed_polygon_path}")
    return watersheds_gdf

# ===================================  

def single_to_multipolygon(polygon_shapefile_path):
    
    gdf = gpd.read_file(polygon_shapefile_path)
    # Group by 'Point_ID' and aggregate geometries into MultiPolygon
    # Check if any 'Point_ID' has duplicates
    if gdf.duplicated(subset=["Point_ID"]).any():
        # Group by 'Point_ID' and aggregate geometries into MultiPolygon
        gdf_multi = gdf.dissolve(by="Point_ID", as_index=False)
    else:
        # No duplicates, keep the original data
        gdf_multi = gdf.copy()
    # Ensure geometries are MultiPolygon
    gdf_multi["geometry"] = gdf_multi["geometry"].apply(
        lambda geom: MultiPolygon([geom]) if geom.geom_type == "Polygon" else geom
    )
    # Save back to the same path (overwrite)
    gdf_multi.to_file(polygon_shapefile_path, driver="ESRI Shapefile")

    print("Shapefile successfully updated with MultiPolygons!")



# -----------------------------------MAIN FUNCTION For preparing watershed characteristics data
def determine_WS_char(
    filtered_watershed_delin_polygon_path,
    filtered_pour_delin_point_path,
    dem_path,
    boundary_polygon_path,
    flow_accum_path,
    prism_30yr_ppt_normals_path,
    cn_table_csv_path,
    usa_states_shapefile_path,
    save_watershed_polygon_path,
    gssurgo_base_raster_path,
    aoi_gSSURGO_hydgrpdcd_data_path,
    NLCD_2024_CONUS_raster_path,
    clipped_nlcd_raster_path,
    temp_folder_path,
    save_reprojected_CN_raster_path,
    base_wetland_data_path,
    pour_ID="Point_ID",user_id=None,
    project_name=None,
    task_type=None,
    check_cancellation_func=None):
    """
    Preparing Data for Region of Influence Calculation

    This function prepares watershed data for Region of Influence (ROI) calculation.

    Steps:

    1. Merge culvert characteristics with filtered watersheds and pour point data.
    2. Calculate watershed mean elevation, slope, and Topographic Wetness Index (TWI).
    3. Calculate watershed average PRISM 30-yr precipitation normals.
    4. Calculate longest channel length.
    5. Calculate maximum overland flow path length.
    6. Calculate time of concentration.
    7. Assign dominant hydrologic soil group from gSSURGO data.
    8. Calculate Curve Number from gSSURGO and NLCD data.

    Parameters:

    - original_pour_point_path (str): Original pour point shapefile path.
    - filtered_watershed_delin_polygon_path (str): Filtered watershed polygon shapefile path.
    - filtered_pour_delin_point_path (str): Filtered pour point shapefile path.
    - dem_path (str): Digital Elevation Model (DEM) raster path.
    - flow_accum_path (str): Flow accumulation raster path.
    - prism_30yr_ppt_normals_path (str): PRISM 30-yr precipitation normals raster path.
    - gssurgo_gdb_path (str): gSSURGO gdb file path.
    - converted_gssurgo_gpkg_path (str): Converted gSSURGO geopackage path.
    - cn_table_csv_path (str): Curve Number table CSV file path.
    - save_watershed_polygon_path (str): Path to save updated watershed polygon shapefile.
    - save_pour_point_path (str): Path to save updated pour point shapefile.
    - aoi_gSSURGO_hydgrpdcd_data_path (str): Path to save gSSURGO soil data.
    - save_nlcd_raster_path (str): Path to save NLCD raster data.
    - temp_save_CN_raster_path (str): Temporary path for CN raster data.
    - save_reprojected_CN_raster_path (str): Path to save reprojected CN raster data.
    - list_gauged_st_pour_ID (list): List of gauged station pour IDs.
    - list_gauged_pt_names (list): List of gauged point names.
    - pour_ID (str): Pour ID column name.
    - list_column_names_to_merge (list): List of column names to merge.

    Returns:

    - geopandas.GeoDataFrame: Updated watershed GeoDataFrame with ROI calculation data.

    Libraries Used:

    - geopandas
    - rasterio
    - numpy
    - pandas
    - zipfile
    """
    try: 
        check_cancellation_func(user_id, project_name, task_type)
        # ___________________ Step 1: Merging culvert characteristics with filtered watersheds and pour point data (these watersheds and pour points are selected for ROI calculations)
        merging_watersheds_and_pour_point_characteristics(
            filtered_watershed_delin_polygon_path,
            filtered_pour_delin_point_path,
            save_watershed_polygon_path,
            pour_ID="Point_ID",
        )
        print("Status: watershed and pour point attributes merged")
        check_cancellation_func(user_id, project_name, task_type)
        # ___________________ Step 2: Calculating mean elevation, mean slope and TWI for each filtered watersheds
        calculate_watershed_mean_elev_slope_TWI(
            dem_path,
            flow_accum_path,
            save_watershed_polygon_path,
            save_watershed_polygon_path,
            WS_ID=pour_ID,
        )
        check_cancellation_func(user_id, project_name, task_type)
        print("Status: watershed mean elevation, mean slope and TWI calculated")

        # ____________________ Step 3: Calculating watershed average values of PRISM 30-yr precipitation normals (1991-2020)
        zonal_average_of_raster(
            prism_30yr_ppt_normals_path,
            dem_path,
            save_watershed_polygon_path,
            save_watershed_polygon_path,
            IDname=pour_ID,
        )
        print("Status: watershed mean 30-yr precipitation normals calculated")
        check_cancellation_func(user_id, project_name, task_type)
        gssurgo_raster_to_clipped_polygon(boundary_polygon_path,
                                        gssurgo_base_raster_path,
                                        aoi_gSSURGO_hydgrpdcd_data_path,
                                        usa_states_shapefile_path
                                        )
        check_cancellation_func(user_id, project_name, task_type)
        assign_dominant_hydrologic_soil_group_to_watersheds_using_GSSURGO(aoi_gSSURGO_hydgrpdcd_data_path,
                                                                            save_watershed_polygon_path,
                                                                        save_watershed_polygon_path)
        print("Status: dominant soil group determined")
        check_cancellation_func(user_id, project_name, task_type) 
        # _____________________ Step 8: Calculating Curve number
        try:
            check_cancellation_func(user_id, project_name, task_type)
            create_cn_raster(
                boundary_polygon_path,
                aoi_gSSURGO_hydgrpdcd_data_path,
                NLCD_2024_CONUS_raster_path,
                cn_table_csv_path,
                temp_folder_path,
                clipped_nlcd_raster_path,
                save_reprojected_CN_raster_path,
            )
            print("Status: watershed Curve Number derived from gSSURGO data and NLCD 2024 data")
        except Exception as e:
            print(f"Unhandled error in CN raster creation: {e}")
            raise
        
        check_cancellation_func(user_id, project_name, task_type)    
        ws_gdf = find_weighted_avg_area_cn_value(
            save_reprojected_CN_raster_path, save_watershed_polygon_path
        )
        print("Status: Dominant Curve Number values assigned to each watershed")

        # _____________________ Step 9: Calculate wetland cover area (%)
        check_cancellation_func(user_id, project_name, task_type)
        wetland_cover_area(
            save_watershed_polygon_path,
            base_wetland_data_path,
            usa_states_shapefile_path,
            save_watershed_polygon_path)
        print("Wetland cover area calculated for each watershed")
        check_cancellation_func(user_id, project_name, task_type)
        single_to_multipolygon(save_watershed_polygon_path)
        print("Status: Data prepared for Region of Influence calculation")
        
        return ws_gdf
    except TaskCancelledError:
            # Re-raise to let the calling function handle it
            raise
    except Exception as e:
        print(f"Error in ws characteristics calculation: {str(e)}")
