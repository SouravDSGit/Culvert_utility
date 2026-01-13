import requests
import zipfile
import os
import geopandas as gpd
import pandas as pd
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
        
def wetland_cover_data(
    polygon_shapefile_path,
    usa_states_shapefile_path,
    save_wetland_polygon_path,
    temp_folder_path
):
    """
    Calculates wetland cover area for a watershed AOI by merging wetland data
    for all US states that the AOI crosses.

    Args:
        polygon_shapefile_path (str): Path to the zipped shapefile containing the boundary polygon.
        usa_states_shapefile_path (str): Path to a known US states shapefile (for state abbreviations).
        save_aoi_wetland_polygon_path (str): Path to save the clipped AOI wetland polygon shapefile.
        save_watershed_polygon_path (str): Path to save the watershed polygon shapefile.
        temp_folder_path (str): Temporary folder to store downloaded files.

    Returns:
        GeoDataFrame: Watershed GeoDataFrame with a new column 'WetAHa' (wetland area in hectares).
    """

    def find_gpkg_file(directory_path):
      """Finds the path to the first .gpkg file in a given directory.
      Args:
        directory_path (str): The path to the directory to search.
      Returns:
        str: The path to the .gpkg file, or None if no file is found.
      """
      for root, dirs, files in os.walk(directory_path):
        for file in files:
          if file.endswith('.gpkg'):
            return os.path.join(root, file)

    # Identify states the AOI intersects with
    state_abv_list = get_us_states_crossed(polygon_shapefile_path, usa_states_shapefile_path)
    print('Selected states:', state_abv_list)
    
    if not state_abv_list:
        print("No states were identified for the given boundary.")
        return None

    wetlands_gdf_list = []

    for state in state_abv_list:
        url = f"https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/{state}_geopackage_wetlands.zip"
        save_wetland_data_path = os.path.join(temp_folder_path,f'{state}_geopackage_wetlands.zip')  ####################### temporary path to save the wetland data for the selected state
        download_data(url, save_wetland_data_path)
            
        unzipped_wetland_folder_path=os.path.splitext(save_wetland_data_path)[0]
        unzip_file(save_wetland_data_path, unzipped_wetland_folder_path)

        def find_gpkg_file(directory_path):
            """Finds the path to the first .gpkg file in a given directory.
            Args:
                directory_path (str): The path to the directory to search.
            Returns:
                str: The path to the .gpkg file, or None if no file is found.
            """
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.endswith('.gpkg'):
                        return os.path.join(root, file)
               
        wetland_gpkg_path = find_gpkg_file(unzipped_wetland_folder_path)
        print('found wetlands gpkg', wetland_gpkg_path) 
        wetlands_gdf = gpd.read_file(wetland_gpkg_path, layer=f'{state}_Wetlands')  # Reading the wetland shapefile
        print('read the state wetland data file')
        wetlands_gdf_aoi = clip_polygon_to_boundary(wetlands_gdf,polygon_shapefile_path)
        print('susccessfully cliped')
        wetlands_gdf_list.append(wetlands_gdf_aoi)
        
    if not wetlands_gdf_list:
        print("No wetland data was successfully loaded.")
        return None

    # Combine all wetlands data
    wetlands_gdf = gpd.GeoDataFrame(pd.concat(wetlands_gdf_list, ignore_index=True), crs=wetlands_gdf_list[0].crs)
    wetlands_gdf["Area_Ha"] = wetlands_gdf["ACRES"] * 0.40468564
    print('calculated area Ha')
    wetlands_gdf.to_file(save_wetland_polygon_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')
    
    
wetland_cover_data(
    polygon_shapefile_path='/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/user_data/1_outputs/WS_deln/Boundary_UTM.shp',
    usa_states_shapefile_path='/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/US States and Territories Shapefile_20250216.zip',
    save_wetland_polygon_path='/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/wetland_polygon.shp',
    temp_folder_path='/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/user_data/1_temp'
)



# def wetland_cover_area(
#     polygon_shapefile_path,
#     usa_states_shapefile_path,
#     save_aoi_wetland_polygon_path,
#     save_watershed_polygon_path,
#     temp_folder_path
# ):
#     """
#     Calculates wetland cover area for a watershed AOI by merging wetland data
#     for all US states that the AOI crosses.

#     Args:
#         polygon_shapefile_path (str): Path to the zipped shapefile containing the boundary polygon.
#         usa_states_shapefile_path (str): Path to a known US states shapefile (for state abbreviations).
#         save_aoi_wetland_polygon_path (str): Path to save the clipped AOI wetland polygon shapefile.
#         save_watershed_polygon_path (str): Path to save the watershed polygon shapefile.
#         temp_folder_path (str): Temporary folder to store downloaded files.

#     Returns:
#         GeoDataFrame: Watershed GeoDataFrame with a new column 'WetAHa' (wetland area in hectares).
#     """

#     def find_gpkg_file(directory_path):
#       """Finds the path to the first .gpkg file in a given directory.
#       Args:
#         directory_path (str): The path to the directory to search.
#       Returns:
#         str: The path to the .gpkg file, or None if no file is found.
#       """
#       for root, dirs, files in os.walk(directory_path):
#         for file in files:
#           if file.endswith('.gpkg'):
#             return os.path.join(root, file)

#     # Identify states the AOI intersects with
#     state_abv_list = get_us_states_crossed(polygon_shapefile_path, usa_states_shapefile_path)
#     print('Selected states:', state_abv_list)
    
#     if not state_abv_list:
#         print("No states were identified for the given boundary.")
#         return None

#     wetlands_gdf_list = []

#     for state in state_abv_list:
#         url = f"https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/{state}_geopackage_wetlands.zip"
#         save_wetland_data_path = os.path.join(temp_folder_path,f'{state}_geopackage_wetlands.zip')  ####################### temporary path to save the wetland data for the selected state
#         download_data(url, save_wetland_data_path)
            
#         unzipped_wetland_folder_path=os.path.splitext(save_wetland_data_path)[0]
#         unzip_file(save_wetland_data_path, unzipped_wetland_folder_path)

#         def find_gpkg_file(directory_path):
#             """Finds the path to the first .gpkg file in a given directory.
#             Args:
#                 directory_path (str): The path to the directory to search.
#             Returns:
#                 str: The path to the .gpkg file, or None if no file is found.
#             """
#             for root, dirs, files in os.walk(directory_path):
#                 for file in files:
#                     if file.endswith('.gpkg'):
#                         return os.path.join(root, file)
               
#         wetland_gpkg_path = find_gpkg_file(unzipped_wetland_folder_path)
#         print('found wetlands gpkg', wetland_gpkg_path) 
#         wetlands_gdf = gpd.read_file(wetland_gpkg_path, layer=f'{state}_Wetlands')  # Reading the wetland shapefile
#         print('read the state wetland data file')
#         wetlands_gdf_aoi = clip_polygon_to_boundary(wetlands_gdf,polygon_shapefile_path)
#         print('susccessfully cliped')
        
#         wetlands_gdf_list.append(wetlands_gdf_aoi)

#         # except Exception as e:
#         #     print(f"Error processing {state}: {e}")
#         #     continue
#         # finally:
#             # Cleanup temporary files
#             # if os.path.exists(save_wetland_data_path):
#             #     os.remove(save_wetland_data_path)
#             # if os.path.exists(unzipped_wetland_folder_path):
#             #     shutil.rmtree(unzipped_wetland_folder_path)

#     if not wetlands_gdf_list:
#         print("No wetland data was successfully loaded.")
#         return None

#     # Combine all wetlands data
#     wetlands_gdf = gpd.GeoDataFrame(pd.concat(wetlands_gdf_list, ignore_index=True), crs=wetlands_gdf_list[0].crs)
#     wetlands_gdf["Area_Ha"] = wetlands_gdf["ACRES"] * 0.40468564
#     print('calculated area Ha')
#     # Iterate over each polygon in the shapefile and clip the GeoDataFrame
#     watersheds_gdf = gpd.read_file(polygon_shapefile_path)
#     for index, row in watersheds_gdf.iterrows():
#         clip_polygon = row['geometry']
#         gdf_clipped = gpd.clip(wetlands_gdf_aoi, clip_polygon)
#         if gdf_clipped.empty:
#            # Calculate the area of wetland cover within each watershed polygon
#            watersheds_gdf.loc[index,'WetAHa'] = 0  ## no wetland cover
#         else:
#            area_Ws = watersheds_gdf.loc[index,'area_ha']
#            # Calculate the area of wetland cover within each watershed polygon
#            watersheds_gdf.loc[index,'WetAHa'] = (gdf_clipped.area*0.0001).sum()  ## *0.0001 converts area in m2 to Ha


#     wetlands_gdf_aoi.to_file(save_aoi_wetland_polygon_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')
#     watersheds_gdf.to_file(save_watershed_polygon_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')
#     return watersheds_gdf
