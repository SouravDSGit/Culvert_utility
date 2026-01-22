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
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import mapping
from shapely.geometry import box
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
import time

# -----------------------------------Function to download and process Hydro Soil Group from GSSURGO data. 
def preprocessing_GSSURGO(cancel_file,
                          zip_gSSURGO_file_path,
                          user_dir_inputs,
                        save_gSSURGO_muaggat_soil_data_path
                        ):
        """
        Assigns the dominant hydrologic soil group to watersheds using gSSURGO data.

        This function performs the following steps:

        1. **Unzips the gSSURGO zip file:**
          - Extracts the gSSURGO data from the provided zip file.
        2. **Converts gSSURGO gdb to geopackage:**
          - Converts the unzipped gSSURGO data from a geodatabase format (.gdb) to a geopackage format (.gpkg) for efficiency.
        3. **Clips gSSURGO data to watershed boundaries:**
          - Extracts the portion of the gSSURGO data that intersects with each watershed boundary.
        4. **Merges relevant layers:**
          - Combines the "Mapunit Aggregated Attribute" (muaggatt) and "MUPOLYGON" layers from the gSSURGO data.
        5. **Saves combined soil attribute shapefile:**
          - Saves the combined "Mapunit Aggregated Attribute" (muaggatt) and "MUPOLYGON" information as a shapefile.

        Parameters:
            zip_gssurgo_file_path (str, optional): Path to the gSSURGO zip file. Defaults to "/content/gSSURGO_SC.zip".
            boundary_polygon_file_path : Path to the boundary shapefile.
            save_gSSURGO_muaggat_soil_data_path (str, optional): Path to save the clipped gSSURGO soil data shapefile. Defaults to "/content/drive/MyDrive/WS_Properties/gSSURGO_data_polygon_UTM_reprojected.shp".

        Returns:
            geopandas.GeoDataFrame: The updated watershed GeoDataFrame with a new column "HySGrpN" containing the numerical representation of the dominant hydrologic soil group.

        Note:
            This function utilizes gSSURGO data and assigns dominant hydrologic soil groups based on the largest area within each watershed.

        Libraries Used:
            - geopandas
            - gdal
            - pandas
            - numpy
        """
        
        """
        Runs the gSSURGO data preprocessing.
        Stops if the cancel file exists.
        """
        try:
            # Simulating the processing loop
            for i in range(100):  # Simulate 100 steps
                if os.path.exists(cancel_file):
                    print("Stopping process: Cancel file detected.")
                    return False  # Indicate process was stopped

                print(f"Processing step {i + 1}...")
                time.sleep(0.5)  # Simulated frequent checking (every 0.5s)

            boundary_path=  os.path.join(user_dir_inputs,'boundary.zip')
            boundary_gdf = gpd.read_file(f"zip://{boundary_path}").to_crs("EPSG:4326")
            # Calculate the centroid of the unified boundary geometry
            centroid = boundary_gdf.geometry.unary_union.centroid

            # Determine the UTM zone from the centroid's longitude
            longitude = centroid.x
            utm_zone = int((longitude + 180) // 6) + 1

            # Construct the UTM CRS based on the calculated zone and hemisphere
            if centroid.y >= 0:
                My_crs = pyproj.CRS(f'EPSG:326{utm_zone}')  # Northern hemisphere
            else:
                My_crs = pyproj.CRS(f'EPSG:327{utm_zone}')  # Southern hemisphere
            boundary_gdf = boundary_gdf.to_crs(My_crs)
            ###########################################################################################################################
            # Creating the gSSURGO geopackage file
            # assigning path to the unzipped gSSURGO data folder
            unzipped_gSSURGO_folder_path = os.path.splitext(zip_gSSURGO_file_path)[0]
            # Get the folder name of the unzipped
            folder_name = os.path.basename(unzipped_gSSURGO_folder_path)
            # Create the new path by appending the folder name [it is required becasue the unzipped folder has nested folder]
            gSSURGO_new_path = os.path.join(unzipped_gSSURGO_folder_path, folder_name)
            gSSURGO_gdb_path = gSSURGO_new_path+'.gdb'
            converted_gssurgo_gpkg_file_path = f'{gSSURGO_new_path}.gdb/gSSURGO.gpkg'

            ###############################################################################################################################

            def clip_mupolygon_to_boundary(vector_data_gdf,boundary_gdf):

                """Clips mupolygon vector data from GSSURGO to the extent of a specified polygon.


                Libraries used:
                - geopandas: For reading, manipulating, and performing spatial operations on geospatial vector data.
                """

                # Load the boundary shapefile
                boundary_gdf["geometry"] = boundary_gdf["geometry"].buffer(0)
                boundary_crs = boundary_gdf.crs
                vector_data_gdf = vector_data_gdf.to_crs(boundary_crs)
                # Clip vector data
                clipped_gdf = gpd.clip(vector_data_gdf, boundary_gdf)

                return clipped_gdf

        
            ####################################################################################################################
            print('gpkg built')
            muaggatt_gdf = gpd.read_file(converted_gssurgo_gpkg_file_path, layer='muaggatt')  # MUAGGATT stands for "Mapunit Aggregated Attribute"


            MUPOLYGON_gdf = gpd.read_file(converted_gssurgo_gpkg_file_path, layer ='MUPOLYGON')  # MUAGGATT stands for "Mapunit Aggregated Attribute"

            aoi_soil_data = pd.merge(muaggatt_gdf,MUPOLYGON_gdf[['MUKEY','geometry']],left_on='mukey',right_on='MUKEY',how='left')
            
            aoi_soil_gdf = gpd.GeoDataFrame(aoi_soil_data,
                                            geometry='geometry_y',
                                            crs=MUPOLYGON_gdf.crs
                                            )

            aoi_soil_gdf_clipped=clip_mupolygon_to_boundary(aoi_soil_gdf,boundary_gdf)
            boundary_gdf["geometry"] = boundary_gdf["geometry"].buffer(0)
            # Drop the 'geometry_x' column
            aoi_soil_gdf_clipped = aoi_soil_gdf_clipped.drop(columns=['geometry_x'])
            aoi_soil_gdf_clipped.to_crs(boundary_gdf.crs).to_file(save_gSSURGO_muaggat_soil_data_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')

            print("Processing completed successfully.")
            return True  # Indicate successful completion

        except Exception as e:
            print(f"Error in preprocessing_GSSURGO: {e}")
            return False  # Indicate failure
        
# # Step 2:
# # --------------------------------------- Function to unzip and save a folder using path
# def unzip_file(file_path, destination_folder):
#   """
#   Unzips a zip file to a specified destination folder.

#   Args:
#       file_path (str): Path to the zip file.
#       destination_folder (str): Path to the folder where the extracted files will be placed.
#   """
#   try:
#     with zipfile.ZipFile(file_path, 'r') as zip_ref:
#       zip_ref.extractall(destination_folder)
#       print(f"Successfully unzipped {file_path} to {destination_folder}")
#   except Exception as e:
#     print(f"Error unzipping {file_path}: {e}")
    
# # # Example usage
# # file_path = '/content/gSSURGO_SC.zip'
# # destination_folder = '/content/gSSURGO_SC'
# # unzip_file(file_path, destination_folder)  

# # -----------------------------------Function to download and process Hydro Soil Group from GSSURGO data. This function also assigns the most dominant (covering most area) soil group for the sub-watersheds.
# def assign_dominant_hydrologic_soil_group_to_watersheds_using_GSSURGO(zip_gSSURGO_file_path,
#                                                                       input_watersheds_path,
#                                                                   output_watersheds_path,
#                                                                   save_gSSURGO_muaggat_soil_data_path
#                                                                   ):
#         """
#         Assigns the dominant hydrologic soil group to watersheds using gSSURGO data.

#         This function performs the following steps:

#         1. **Unzips the gSSURGO zip file:**
#           - Extracts the gSSURGO data from the provided zip file.
#         2. **Converts gSSURGO gdb to geopackage:**
#           - Converts the unzipped gSSURGO data from a geodatabase format (.gdb) to a geopackage format (.gpkg) for efficiency.
#         3. **Clips gSSURGO data to watershed boundaries:**
#           - Extracts the portion of the gSSURGO data that intersects with each watershed boundary.
#         4. **Merges relevant layers:**
#           - Combines the "Mapunit Aggregated Attribute" (muaggatt) and "MUPOLYGON" layers from the gSSURGO data.
#         5. **Finds dominant hydrologic soil group for each watershed:**
#           - Analyzes the clipped gSSURGO data for each watershed and identifies the hydrologic soil group with the largest area within that watershed boundary.
#         6. **Assigns dominant group to watershed GeoDataFrame:**
#           - Updates the watershed GeoDataFrame by adding a new column named "HySGrp" containing the dominant hydrologic soil group for each watershed.
#         7. **Converts HySGrp values and assigns numerical values:**
#           - Converts any values in the "HySGrp" column containing '/' to 'D'.
#           - Assigns numerical values (1 to 4) to the different hydrologic soil groups (A, B, C, D).
#         8. **Saves updated watershed shapefile:**
#           - Saves the updated watershed GeoDataFrame containing the dominant hydrologic soil group information as a shapefile.

#         Parameters:
#             zip_gssurgo_file_path (str, optional): Path to the gSSURGO zip file. Defaults to "/content/gSSURGO_SC.zip".
#             save_watershed_file_path (str, optional): Path to the watershed shapefile. Defaults to "/content/drive/MyDrive/WS_Properties/filtered_watersheds_by_area_UTM_reprojected.shp".
#             save_gSSURGO_muaggat_soil_data_path (str, optional): Path to save the clipped gSSURGO soil data shapefile. Defaults to "/content/drive/MyDrive/WS_Properties/gSSURGO_data_polygon_UTM_reprojected.shp".

#         Returns:
#             geopandas.GeoDataFrame: The updated watershed GeoDataFrame with a new column "HySGrpN" containing the numerical representation of the dominant hydrologic soil group.

#         Note:
#             This function utilizes gSSURGO data and assigns dominant hydrologic soil groups based on the largest area within each watershed.

#         Libraries Used:
#             - geopandas
#             - gdal
#             - pandas
#             - numpy
#         """

#         ###########################################################################################################################
#         # Creating the gSSURGO geopackage file
#         # assigning path to the unzipped gSSURGO data folder
#         unzipped_gSSURGO_folder_path = os.path.splitext(zip_gSSURGO_file_path)[0]
#         # unzipping the gSSURGO data and saving it to the assigned path
#         unzip_file(zip_gSSURGO_file_path, unzipped_gSSURGO_folder_path)
#         # Get the folder name of the unzipped
#         folder_name = os.path.basename(unzipped_gSSURGO_folder_path)
       
#         # Create the new path by appending the folder name [it is required becasue the unzipped folder has nested folder]
#         gSSURGO_new_path = os.path.join(unzipped_gSSURGO_folder_path, folder_name)
        
#         # gSSURGO_gdb_path = next((os.path.join(root, folder) for root, dirs, _ in os.walk(gSSURGO_new_path)
#         #                         for folder in dirs if folder.endswith('.gdb')), None)
#         gSSURGO_gdb_path = gSSURGO_new_path+'.gdb'
#         converted_gssurgo_gpkg_file_path = f'{gSSURGO_new_path}.gdb/gSSURGO.gpkg'
        
#         # function to convert .gdb to .gpkg
#         def build_gpkg(gssurgo_gdb_path, converted_gssurgo_gpkg_path):
#             """Build a geopackage from gSSURGO source gdb data."""
#             src_ds = gdal.OpenEx(gssurgo_gdb_path)
#             ds = gdal.VectorTranslate(converted_gssurgo_gpkg_path, srcDS = src_ds, format = "GPKG")
#             del ds

#         ###############################################################################################################################

#         def clip_mupolygon_to_boundary(vector_data_gdf,boundary_polygon_file_path):

#             """Clips mupolygon vector data from GSSURGO to the extent of a specified polygon.


#             Libraries used:
#             - geopandas: For reading, manipulating, and performing spatial operations on geospatial vector data.
#             """

#             # Load the boundary shapefile
#             boundary_gdf = gpd.read_file(boundary_polygon_file_path)
#             boundary_gdf["geometry"] = boundary_gdf["geometry"].buffer(0)
#             boundary_crs = boundary_gdf.crs
#             vector_data_gdf = vector_data_gdf.to_crs(boundary_crs)
#             # Clip vector data
#             clipped_gdf = gpd.clip(vector_data_gdf, boundary_gdf)

#             return clipped_gdf

#         # Function to clip a GeoDataFrame with a polygon and find the dominant group
#         def find_dominant_soil_group(aoi_soil_gdf_clipped, clip_polygon):
#             gdf_clipped = gpd.clip(aoi_soil_gdf_clipped, clip_polygon)
#             if gdf_clipped.empty:
#                 return None

#             # Calculate the area of each clipped polygon
#             gdf_clipped['area'] = gdf_clipped.area

#             # Find the dominant group based on the largest area
#             dominant_group = gdf_clipped.loc[gdf_clipped['area'].idxmax(), 'hydgrpdcd']
#             return dominant_group

#         # Function to convert values containing '/' to 'D'
#         def convert_values(value):
#             if isinstance(value, str) and '/' in value:
#                 return 'D'
#             else:
#                 return value

#         ####################################################################################################################
#         build_gpkg(gSSURGO_gdb_path,converted_gssurgo_gpkg_file_path)
#         print('gpkg built')
#         muaggatt_gdf = gpd.read_file(converted_gssurgo_gpkg_file_path, layer='muaggatt')  # MUAGGATT stands for "Mapunit Aggregated Attribute"


#         MUPOLYGON_gdf = gpd.read_file(converted_gssurgo_gpkg_file_path, layer ='MUPOLYGON')  # MUAGGATT stands for "Mapunit Aggregated Attribute"

#         aoi_soil_data = pd.merge(muaggatt_gdf,MUPOLYGON_gdf[['MUKEY','geometry']],left_on='mukey',right_on='MUKEY',how='left')
        
#         aoi_soil_gdf = gpd.GeoDataFrame(aoi_soil_data,
#                                         geometry='geometry_y',
#                                         crs=MUPOLYGON_gdf.crs
#                                         )

#         aoi_soil_gdf_clipped=clip_mupolygon_to_boundary(aoi_soil_gdf,input_watersheds_path)
#         # Drop the 'geometry_x' column
#         aoi_soil_gdf_clipped = aoi_soil_gdf_clipped.drop(columns=['geometry_x'])
        
#         print('clip successful')
#         # Iterate over each polygon in the shapefile and clip the GeoDataFrame
#         watersheds_gdf = gpd.read_file(input_watersheds_path)
#         aoi_soil_gdf_clipped.to_crs(watersheds_gdf.crs).to_file(save_gSSURGO_muaggat_soil_data_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')
#         print('clipped soil data saved')
#         for index, row in watersheds_gdf.iterrows():
#             clip_polygon = row['geometry']
#             dominant_group = find_dominant_soil_group(aoi_soil_gdf_clipped, clip_polygon)
#             watersheds_gdf.loc[index,'HySGrp'] = dominant_group
#         # print(f"before_converting {watersheds_gdf['HySGrp'].unique()}")


#         # Apply the function to the 'Column2'
#         watersheds_gdf['HySGrp'] = watersheds_gdf['HySGrp'].apply(convert_values)
#         # print(f"after_converting {watersheds_gdf['HySGrp'].unique()}")
#         watersheds_gdf['HySGrpN']=np.nan
#         watersheds_gdf.loc[watersheds_gdf['HySGrp']=='A','HySGrpN']=1
#         watersheds_gdf.loc[watersheds_gdf['HySGrp']=='B','HySGrpN']=2
#         watersheds_gdf.loc[watersheds_gdf['HySGrp']=='C','HySGrpN']=3
#         watersheds_gdf.loc[watersheds_gdf['HySGrp']=='D','HySGrpN']=4
#         watersheds_gdf.to_file(output_watersheds_path,driver='ESRI Shapefile',encoding='utf-8', mode='w')

#         return watersheds_gdf

