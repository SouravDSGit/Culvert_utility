import zipfile
import os
import shutil
import geopandas as gpd

def process_shapefile_zip(input_zipped_file_path, user_dir_temp, output_path, merge=True):
   required_extensions = ['.shp', '.shx', '.dbf', '.prj']
   
   # Check if required files exist in zip
   with zipfile.ZipFile(input_zipped_file_path, 'r') as zip_ref:
       zip_files = [f.lower() for f in zip_ref.namelist()]
       if not all(any(f.endswith(ext) for f in zip_files) for ext in required_extensions):
           raise ValueError("Required shapefile components missing")
       
       # Extract only required files to temporary directory
       for file in zip_ref.namelist():
           if any(file.lower().endswith(ext) for ext in required_extensions):
               zip_ref.extract(file, user_dir_temp)
   
   # Find the .shp file
   shp_file = None
   for file in os.listdir(user_dir_temp):
       if file.endswith('.shp'):
           shp_file = os.path.join(user_dir_temp, file)
           break
   
   if merge:
       # Read, merge polygons, and save
       gdf = gpd.read_file(shp_file)
       merged_polygon = gdf.union_all
       merged_gdf = gpd.GeoDataFrame([1], geometry=[merged_polygon], crs=gdf.crs)
       
       # Save merged shapefile to temp location
       temp_merged_dir = os.path.join(user_dir_temp, 'merged')
       os.makedirs(temp_merged_dir, exist_ok=True)
       temp_merged_shp = os.path.join(temp_merged_dir, 'merged.shp')
       merged_gdf.to_file(temp_merged_shp)
       
       # Create zip from merged files
       with zipfile.ZipFile(output_path, 'w') as new_zip:
           for file in os.listdir(temp_merged_dir):
               if any(file.lower().endswith(ext) for ext in required_extensions):
                   file_path = os.path.join(temp_merged_dir, file)
                   new_zip.write(file_path, file)
   else:
       # Create new zip with only required files (original behavior)
       with zipfile.ZipFile(output_path, 'w') as new_zip:
           for root, dirs, files in os.walk(user_dir_temp):
               for file in files:
                   if any(file.lower().endswith(ext) for ext in required_extensions):
                       file_path = os.path.join(root, file)
                       new_zip.write(file_path, file)
   
   # Clean up temporary files
#    shutil.rmtree(user_dir_temp, ignore_errors=True)