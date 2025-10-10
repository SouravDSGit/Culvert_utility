# utils/subroutine_watershed_delineation.py
#=================================================================================================
# Libraries
#=================================================================================================
# For analysis
import numpy as np
import pandas as pd
import geopandas as gpd
import whitebox
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
import fiona
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union, nearest_points
import pyproj
import osmnx as ox
import os
import json
import folium
from folium.plugins import MeasureControl, Draw, Fullscreen
import shutil, time

from app import TaskCancelledError
# Initializing whitebox tools
wbt = whitebox.WhiteboxTools()
from static.visualization.subroutine_basemap_generator import generate_basemap_results 
#=================================================================================================
# Function to determine the UTM coordinate projection system
#=================================================================================================
def get_utm_crs_from_wgs84(Boundary_poly_path):
    """
    Determines the appropriate UTM (Universal Transverse Mercator) coordinate reference system (CRS)
    for a given polygon boundary shapefile based on its centroid's geographic location.

    This function reads a boundary shapefile in WGS84 (EPSG:4326), calculates the centroid,
    and identifies the corresponding UTM zone. It then constructs the UTM CRS, differentiating
    between the Northern and Southern Hemispheres.

    Parameters:
    - Boundary_poly_path (str): Path to the input boundary shapefile.

    Returns:
    - My_crs (pyproj.CRS): The UTM CRS corresponding to the centroid's location, suitable for mapping
      and geospatial analysis.

    Libraries used:
    - geopandas: For reading and manipulating geospatial data.
    - pyproj: For handling coordinate reference systems and transformations.
    """

    try:
        # Check if the file is in a .zip archive
        if Boundary_poly_path.lower().endswith('.zip'):
            reg_gdf = gpd.read_file(f"zip://{Boundary_poly_path}").to_crs("EPSG:4326")
        else:
            reg_gdf = gpd.read_file(Boundary_poly_path).to_crs("EPSG:4326")

        # Ensure the GeoDataFrame is not empty
        if reg_gdf.empty:
            raise ValueError("The boundary shapefile contains no data.")

        # Calculate the centroid of the unified boundary geometry
        centroid = reg_gdf.geometry.unary_union.centroid

        # Determine the UTM zone from the centroid's longitude
        longitude = centroid.x
        utm_zone = int((longitude + 180) // 6) + 1

        # Construct the UTM CRS based on the calculated zone and hemisphere
        if centroid.y >= 0:
            My_crs = pyproj.CRS(f'EPSG:326{utm_zone}')  # Northern hemisphere
        else:
            My_crs = pyproj.CRS(f'EPSG:327{utm_zone}')  # Southern hemisphere

        return My_crs
    except Exception as e:
        raise RuntimeError(f"Error processing the boundary shapefile: {e}")

#========================================================================================================================================
#Function to project the vector data into UTM coordinates
#=========================================================================================================================================
def project_vector_data_to_utm(input_vector_data_path,output_vector_data_path):
  """
    Reprojects geospatial vector data (e.g., polygons, points) to the appropriate Universal Transverse Mercator (UTM)
    coordinate reference system (CRS) based on its geographic location.

    This function:
    1. **Reads Input Data:** Reads the input vector data from the specified path in a geographic coordinate system (GCS), typically WGS84 (EPSG:4326).
    2. **Determines UTM CRS:** Calculates the appropriate UTM CRS based on the geographic extent of the input data.
    3. **Reprojects Data:** Reprojects the input data from the GCS to the determined UTM CRS.
    4. **Writes Output Data:** Writes the reprojected data to the specified output path in the UTM CRS.

    **Parameters:**
    - `input_vector_data_path` (str): Path to the input vector data file (e.g., shapefile, GeoJSON).
    - `output_vector_data_path` (str): Path to the output vector data file.

    **Returns:**
    - None: The function directly writes the reprojected data to the output path.

    **Libraries Used:**
    - `geopandas`: For reading, manipulating, and writing geospatial data.
    - `pyproj`: For handling coordinate reference systems and transformations.
    """
  try:
    utm_crs=get_utm_crs_from_wgs84(input_vector_data_path)
    if input_vector_data_path.lower().endswith('.zip'):
        input_gdf = gpd.read_file(f"zip://{input_vector_data_path}")
    else:
        input_gdf = gpd.read_file(input_vector_data_path)

    input_gdf.to_crs(utm_crs).to_file(output_vector_data_path)
  except Exception as e:
        raise RuntimeError(f"Error processing the vector data: {e}")

#================================================================================================================================
#Function to project raster data into UTM coordinates
#=================================================================================================================================
def reproject_raster_from_path(input_path, output_path, dst_crs):
    """
    Reprojects a raster file to a specified coordinate reference system (CRS).

    This function reads an input raster file, computes the necessary transformations to reproject
    it to a specified CRS, and saves the reprojected raster to a new file. It handles multiple bands
    in the raster and retains the original metadata while updating the CRS and transform information.

    Args:
        input_path (str): Path to the input raster file.
        output_path (str): Path to the output reprojected raster file.
        dst_crs (str): The destination CRS to which the raster will be reprojected.

    Returns:
        None: The function saves the reprojected raster to the specified output path.

    Libraries used:
    - rasterio: For reading and writing raster data and handling coordinate reference systems.
    - rasterio.warp: For performing the reprojection of the raster bands.
    - rasterio.enums: For handling resampling methods.
    """

    with rasterio.open(input_path) as src:
        # Calculate transform matrix for output
        src_transform = src.transform
        dst_transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "nodata": 0,  # Replace 0 with np.nan if needed
            }
        )

        with rasterio.open(output_path, "w", **dst_kwargs) as dst:
            # Iterate through bands
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

#================================================================================================================================
#Function to clip raster data to the region boundary with offset in meters outside the boundary
#================================================================================================================================
def clip_raster_with_offset(polygon_path, raster_path, output_raster_path, offset_distance_m=150):
    """Clips a raster with an offset around a polygon.

    This function reads a polygon shapefile and an input raster file, creates a buffer (or offset)
    around the polygon, and clips the raster to the extent of the buffered polygon. The resulting
    clipped raster is saved to a specified output path. This is useful for focusing on a specific
    area of interest in a raster dataset while including additional context defined by the offset.

    Args:
        polygon_path (str): Path to the polygon shapefile defining the area of interest.
        raster_path (str): Path to the input raster file that will be clipped.
        output_raster_path (str): Path to save the clipped raster.
        offset_distance_m (float): Width of the buffer around the polygon in meters. Default is 150 meters.

    Returns:
        None: The function saves the clipped raster to the specified output path.

    Libraries used:
    - geopandas: For reading and manipulating geospatial vector data (polygons).
    - rasterio: For reading and writing raster data and performing operations like masking.
    - rasterio.mask: For masking the raster with the buffered polygon.
    """

    # Open the polygon shapefile
    gdf = gpd.read_file(polygon_path)

    # Ensure the polygon is in the same CRS as the raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Reproject the polygon to the raster CRS
    gdf = gdf.to_crs(raster_crs)

    # Create a buffer around the polygon
    gdf['geometry'] = gdf.buffer(offset_distance_m)

    # Convert the expanded polygon to GeoJSON format
    buffered_polygons = [geom for geom in gdf['geometry']]

    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Mask the raster with the expanded polygon
        out_image, out_transform = mask(src, buffered_polygons, crop=True)

        # Update metadata to reflect the new dimensions and transform
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save the clipped raster to the output path
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

#==============================================================================================================================================
#Function to clip vector files to boundary polygon
#==============================================================================================================================================
def clip_vector_data_to_polygon(vector_data_file_path,polygon_file_path,save_clipped_vector_file_path):

    """Clips vector data to the extent of a specified polygon.

    This function reads vector data from a specified file and clips it to the area defined by a
    polygon shapefile. The resulting clipped vector data is saved to a specified output file.
    This is useful for extracting specific features within a certain boundary or area of interest.

    Args:
        vector_data_file_path (str): Path to the vector data file (e.g., shapefile) to be clipped.
        polygon_file_path (str): Path to the polygon shapefile that defines the clipping boundary.
        save_clipped_vector_file_path (str): Path to save the resulting clipped vector data.

    Returns:
        None: The function saves the clipped vector data to the specified output file.

    Libraries used:
    - geopandas: For reading, manipulating, and performing spatial operations on geospatial vector data.
    """

    # Load the boundary shapefile
    boundary_gdf = gpd.read_file(polygon_file_path)
    # load data to be clipped
    gdf = gpd.read_file(vector_data_file_path)
    # check if the pour points are grouped then find the unique pour point for ws deln
    if 'Grp_ID' in gdf.columns:
         non_na_gdf = gdf[gdf['Grp_ID'].notna()] # rows with non NA values
         na_gdf = gdf[gdf['Grp_ID']=='NA'] # rows with NA values

         gdf_unique = non_na_gdf.drop_duplicates(subset='Grp_ID',keep='first')
         result_gdf = pd.concat([gdf_unique,na_gdf])
    else:
         result_gdf = gdf
    # Clip vector data
    clipped_gdf = gpd.clip(result_gdf, boundary_gdf)
    clipped_gdf.to_file(save_clipped_vector_file_path)

#==================================================================================================================================
#Function to download open street map or road data for the boundary region with a buffer in meters
#==================================================================================================================================
def download_osm_roads_with_buffer(Boundary_poly_path, save_road_data_path):
    """
    Downloads OpenStreetMap road data within a buffered boundary area, infers road widths, and saves the data.

    Parameters:
    - EFR_boundary_path (str): Path to the input boundary shapefile.
    - save_road_data_path (str): Path to save the output road data shapefile.
    - offset_distance_m (float): Offset distance in meters for expanding the boundary. Default is 150 meters.
    Libraries used:
    - import geopandas as gpd
    - import osmnx as ox
    - from pyproj import Transformer
    Returns:
    - GeoDataFrame of roads with inferred widths for visualization and analysis.
    """
    utm_crs = get_utm_crs_from_wgs84(Boundary_poly_path)
    boundary_gdf = gpd.read_file(Boundary_poly_path).to_crs(utm_crs)
    # offset_distance = meters_to_degrees(boundary_gdf, distance_meters=offset_distance_m)
    # print('offset_in_degrees: ',offset_distance)
    ## Creating Buffered Bounding Box
    offset_distance = 0.025  # in latitude, logitude degrees
    min_lat, min_lon, max_lat, max_lon = boundary_gdf.to_crs('EPSG:4326').total_bounds
    offset_bbox = [
        min_lat - offset_distance,
        min_lon - offset_distance,
        max_lat + offset_distance,
        max_lon + offset_distance
    ]

    G = ox.graph_from_bbox(
        offset_bbox[3],  # north
        offset_bbox[1],  # south
        offset_bbox[2],  # east
        offset_bbox[0],  # west
        network_type='drive'
    )

    roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Infer road width based on highway type
    width_by_type = {'motorway': 25, 'trunk': 15, 'primary': 12, 'secondary': 10, 'tertiary': 8, 'residential': 6, 'unclassified': 5}
    roads_gdf['road_width'] = roads_gdf['highway'].apply(lambda hwy: width_by_type.get(hwy[0] if isinstance(hwy, list) else hwy, None))

    # Save road data in UTM coordinates
    roads_gdf.to_crs(utm_crs).to_file(save_road_data_path)
    return roads_gdf


#==================================================================================================================================
#Function to snap point features to polyline feature within a snap distance in meters
#==================================================================================================================================
def snap_points_to_polyline(points_path, polyline_path, snapped_point_path, ID_column, snap_distance_m=10):
    """Snaps points to the nearest polyline within a specified distance.

    This function reads a set of points and polylines from specified file paths, and snaps each
    point to the nearest polyline if it is within a given snapping distance. The snapped points
    are then saved to a new shapefile, including an identifier from the original points.

    Args:
        points_path (str): Path to the input points shapefile.
        polyline_path (str): Path to the input polyline shapefile.
        snapped_point_path (str): Path to save the output snapped points shapefile.
        ID_column (str): The name of the column containing unique identifiers in the points dataset.
        snap_distance_m (float): The distance (in the units of the coordinate reference system)
                               within which to snap points to the nearest polyline. Default is 10 m.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the snapped points and their associated identifiers.

    Raises:
        ValueError: If ID_column is not a string.

    Libraries used:
    - geopandas: For reading and manipulating geospatial vector data.
    - shapely: For geometric operations, including buffering and distance calculations.
    - scipy.spatial: For finding the nearest point on a polyline.

    Notes:
    - The function assumes that the input coordinate systems for the points and polylines are the same.
    - The snapping is performed in geographic coordinates, meaning the units are based on the CRS (e.g., meters if UTM).
    """

    if not isinstance(ID_column, str):
        raise ValueError("ID_column must be a string. Provide pour point ID column name.")
    # Load points and polyline shapefiles
    points = gpd.read_file(points_path)
    polylines = gpd.read_file(polyline_path)

    # Snap points in geographic coordinates, not pixels
    snapped_points = []
    fid=1
    for idx, point in points.iterrows():
        # Buffer the point by the snap_distance (in geographic space, i.e., meters if UTM)
        buffered_point = point['geometry'].buffer(snap_distance_m)

        # Query the spatial index of the polyline (in geographic coordinates)
        possible_matches_index = list(polylines.sindex.intersection(buffered_point.bounds))

        # Filter possible polylines that intersect with the buffer
        possible_matches = polylines.iloc[possible_matches_index]

        # Check if any polylines are nearby
        if not possible_matches.empty:
            # Find the nearest polyline to the point
            nearest_polyline = possible_matches.geometry.distance(point['geometry']).idxmin()
            nearest_polyline_geom = possible_matches.loc[nearest_polyline, 'geometry']

            # Find the nearest point on the polyline to the point
            snapped_point = nearest_points(point['geometry'], nearest_polyline_geom)[1]
        else:
            # If no nearby polylines, keep the original point
            snapped_point = point['geometry']

        # snapped_points.append(snapped_point)
        snapped_points.append({
              'geometry': snapped_point,
              'FID' : fid,
              ID_column: point[ID_column]  # Assuming the original points have an 'FID' column
              })
        fid +=1

    # Create a GeoDataFrame for the snapped points
    snapped_gdf = gpd.GeoDataFrame(snapped_points, crs=points.crs)

    # Save the snapped points to a new shapefile
    snapped_gdf.to_file(snapped_point_path)

    return snapped_gdf


#==================================================================================================================================
#Function to create breakline polyline features and save them to a shapefile
#==================================================================================================================================
def create_breaklines(polyline_path, points_file_path, output_breakline_file_path, offset=2, road_data_uploaded_by_user=None):

    """Generates breaklines based on the nearest road segments to a set of points.

    This function reads road polyline and point data from specified file paths, projects each point
    onto the nearest road segment, and creates perpendicular breaklines with specified offsets.
    The generated breaklines are then saved to a new shapefile.

    Args:
        polyline_path (str): Path to the input polyline shapefile containing road segments.
        points_file_path (str): Path to the input point shapefile containing points for which breaklines will be created.
        output_breakline_file_path (str): Path to save the output breakline shapefile.
        offset (float, optional): Distance to offset the breaklines from the road segments, in the same units as the CRS.
                                   Default is 2 meters.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the generated breaklines.

    Libraries used:
    - geopandas: For reading and manipulating geospatial vector data.
    - shapely: For geometric operations, including projections and line segment creation.
    - numpy: For numerical operations and vector manipulations.

    Notes:
    - The function assumes that both the input point and polyline shapefiles are in the same coordinate reference system (CRS).
    - Breaklines are created as perpendicular segments to the road at each projected point, with specified offsets to account for road width.
    """

    # Load road and points data
    road_gdf = gpd.read_file(polyline_path)
    points_gdf = gpd.read_file(points_file_path)
    #segment_length (float): Length of the line segments (in meters) to which perpendicular breaklines will be created.
    segment_length=1
    # Store the generated line segments and perpendicular segments
    cut_segments = []
    breaklines = []

    # Iterate through each point
    for _, point in points_gdf.iterrows():
        point_geom = point.geometry

        # Find the nearest road segment to the point
        nearest_road = road_gdf.geometry.unary_union


        if nearest_road.is_empty:
            continue

        # Project the point onto the road to find the closest point on the road
        projected_point = nearest_road.interpolate(nearest_road.project(point_geom))

        # Create the direction vector from the projected point to the next point on the road
        if nearest_road.distance(point_geom) == 0:  # Point is on the road
            road_point = point_geom
        else:
            road_point = projected_point

        # Get the nearest road segment to the projected point
        segment_index = road_gdf.distance(road_point).idxmin()
        selected_line = road_gdf.geometry.iloc[segment_index]
    
        # Determine road width
        if road_data_uploaded_by_user == 1:
            if 'road_width' in road_gdf.columns:
                width_value = road_gdf.loc[segment_index, 'road_width']
                try:
                    nearest_road_width = float(width_value) if pd.notna(width_value) else 10.0
                except:
                    nearest_road_width = 10.0
            else:
                nearest_road_width = 10.0
        else:
            # Use OSM-based known road width column
            nearest_road_width = road_gdf.road_width[segment_index]

        # Create a tangent segment at the projected point
        if selected_line.geom_type == 'MultiLineString':
            selected_line = selected_line.geoms[0]  # Use first component
        coords = np.array(selected_line.coords)
        direction = (coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])

        # Normalize the direction vector
        length = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        if length == 0:  # Avoid division by zero
            continue

        direction_normalized = (direction[0] / length, direction[1] / length)

        # Create left and right endpoints for the cut segment
        left_segment_start = (road_point.x - (segment_length / 2) * direction_normalized[1],
                              road_point.y + (segment_length / 2) * direction_normalized[0])
        left_segment_end = (road_point.x + (segment_length / 2) * direction_normalized[1],
                            road_point.y - (segment_length / 2) * direction_normalized[0])

        right_segment_start = (road_point.x + (segment_length / 2) * direction_normalized[1],
                               road_point.y - (segment_length / 2) * direction_normalized[0])
        right_segment_end = (road_point.x - (segment_length / 2) * direction_normalized[1],
                             road_point.y + (segment_length / 2) * direction_normalized[0])

        cut_segments.append(LineString([left_segment_start, left_segment_end]))
        cut_segments.append(LineString([right_segment_start, right_segment_end]))

        # Create the midpoint for the perpendicular segment
        midpoint = Point(road_point.x, road_point.y)

        # Perpendicular vector (rotate 90 degrees)
        perp_vector = (-direction_normalized[1], direction_normalized[0])  # Rotate 90 degrees
        perpendicular_length = nearest_road_width + offset
        # Create the perpendicular line segment
        start_point = (midpoint.x + (perpendicular_length / 2) * perp_vector[0],
                       midpoint.y + (perpendicular_length / 2) * perp_vector[1])
        end_point = (midpoint.x - (perpendicular_length / 2) * perp_vector[0],
                     midpoint.y - (perpendicular_length / 2) * perp_vector[1])
        perpendicular_segment = LineString([start_point, end_point])
        breaklines.append(perpendicular_segment)
        # Create a GeoDataFrame for the perpendicular segments
    breakline_segments_gdf = gpd.GeoDataFrame(geometry=breaklines, crs=points_gdf.crs)

    # Save the GeoDataFrame to a shapefile
    breakline_segments_gdf.to_file(output_breakline_file_path)
    return breakline_segments_gdf

#===================================================================================================================================
#Function to adjust DEM based on poyline (whether to burn or fill DEM) for hydro-enforement
#===================================================================================================================================
def adjust_dem_along_polyline(dem_UTM_path, polyline_UTM_path, adjusted_dem_UTM_path, dy, burn, buffer_width, target_crs):
    """
    Adjusts the elevation of a Digital Elevation Model (DEM) raster along specified road segments,
    either by burning (decreasing) or filling (increasing) the DEM elevation in areas around
    buffered road segments.

    Parameters:
        dem_UTM_path (str): Path to the input DEM raster file in UTM coordinate system.
        polyline_UTM_path (str): Path to the input polyline vector shapefile representing road segments, in UTM.
        adjusted_dem_UTM_path (str): Path to save the adjusted DEM raster file in UTM.
        dy (float): The increment or decrement value to adjust the DEM elevation.
                    Positive values increase elevation (fill), while negative values decrease elevation (burn).
        burn (bool): If True, burn (lower) the DEM by decreasing elevation. If False, fill (increase) the DEM.
        buffer_width (float): The buffer width around the polyline vector (in meters).
                              For example, if buffer_width = 2, the DEM will be adjusted
                              within a 2-meter offset on each side of the polyline.
        target_crs: The target coordinate reference system (CRS) for the adjusted DEM.

    Returns:
        None: The function saves the adjusted DEM raster to the specified file path.

    Libraries used:
    - rasterio: For reading and writing raster data and handling raster metadata.
    - fiona: For reading vector data from shapefiles.
    - shapely: For geometric operations, including creating and buffering geometries.
    - numpy: For numerical operations and handling array manipulations.
    - rasterstats: For rasterization and manipulating raster data based on vector geometries.

    Notes:
    - The function creates a buffer around the road segments, and only the DEM pixels within this buffer
      are adjusted based on the specified dy value and burn/fill operation.
    - The target CRS should be specified to ensure the output DEM is correctly aligned with other spatial data.
    """
    # Set the burn value depending on whether we're burning or filling
    burn_by = -1 * dy if burn else dy

    # Read the DEM raster
    with rasterio.open(dem_UTM_path) as dem_src:
        dem = dem_src.read(1)
        dem_meta = dem_src.meta
        nodata = dem_src.nodata

    # Read the road shapefile
    with fiona.open(polyline_UTM_path) as road_src:
        road = [shape(feature['geometry']) for feature in road_src]

    # Create a buffer around the road polylines
    buffered_road = [road.buffer(buffer_width) for road in road]
    buffered_union = unary_union(buffered_road)

    # Create a mask where the buffered roads are located
    mask = rasterize(
        [(geom, 1) for geom in buffered_road],
        out_shape=dem.shape,
        transform=dem_meta['transform'],
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8
    )

    # Adjust the DEM elevation, excluding nodata values
    dem_adjusted = np.where((mask == 1) & (dem != nodata), dem + burn_by, dem)

    # Update the metadata to reflect the nodata value
    dem_meta.update(nodata=nodata)

    # Calculate the transform and dimensions for the target CRS
    transform, width, height = calculate_default_transform(
        dem_meta['crs'], target_crs, dem_meta['width'], dem_meta['height'], *dem_src.bounds
    )

    # Update metadata for the new DEM
    new_dem_meta = dem_meta.copy()
    new_dem_meta.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Write the adjusted DEM to a new file
    with rasterio.open(adjusted_dem_UTM_path, 'w', **new_dem_meta) as dst:
        dst.write(dem_adjusted, 1)

    print(f"Adjusted DEM saved at {adjusted_dem_UTM_path}")

#=======================================================================================================================================
#Function to sanp points to nearest point feature within a snap distance
#========================================================================================================================================
def snap_points_to_nearest_points_within_distance(points_path, target_points_path, output_path, ID_column, snap_distance):
    """
    Snap each point from the points shapefile to the nearest point in the target points shapefile
    if within a specified snap distance, preserving the original IDs in the output. Points that
    are further than the specified snap distance from any target point are excluded from the output.

    Parameters
    ----------
    points_path : str
        Path to the shapefile containing the points to be snapped.
    target_points_path : str
        Path to the shapefile containing the target points to snap to.
    output_path : str
        Path to save the output shapefile of snapped points.
    ID_column : str
        The name of the column containing IDs in the original points shapefile, which will be retained
        in the output.
    snap_distance : float
        The maximum distance within which points will be snapped to the nearest target point.

    Returns
    -------
    snapped_points : list
        A list of dictionaries containing snapped point geometries and original IDs, or an empty list
        if no points were snapped.

    Raises
    ------
    ValueError
        If ID_column is not a string.

    Notes
    -----
    - The function first checks if both input shapefiles share the same coordinate reference system
      (CRS) and reprojects the target points if necessary.
    - Each point from the input shapefile is checked against the target points to find the nearest
      point. If the nearest point is within the specified snap distance, it is added to the output.
    - The function uses the Shapely library to calculate distances and find nearest points.

    Example
    -------
    To use this function, provide the paths to the points and target points shapefiles, specify
    the output path for the snapped points, and define the ID column and snap distance. For example:

    snap_points_to_nearest_points_within_distance('path/to/points.shp',
                                                  'path/to/target_points.shp',
                                                  'path/to/output_snapped_points.shp',
                                                  'PointID',
                                                  10.0)
    """
    if not isinstance(ID_column, str):
        raise ValueError("ID_column must be a string. Provide pour point ID column name.")

    # Load the points and target points from shapefiles
    points_gdf = gpd.read_file(points_path)
    target_points_gdf = gpd.read_file(target_points_path)

    # Ensure both GeoDataFrames are in the same CRS
    if points_gdf.crs != target_points_gdf.crs:
        target_points_gdf = target_points_gdf.to_crs(points_gdf.crs)

    # List to store the snapped points with original IDs
    snapped_points = []
    fid =1
    # Loop through each point to snap
    for idx, point in points_gdf.iterrows():
        original_point = point.geometry

        # Find the nearest point in the target GeoDataFrame
        nearest_geom = nearest_points(original_point, target_points_gdf.unary_union)[1]

        # Calculate the distance to the nearest point
        distance_to_nearest = original_point.distance(nearest_geom)

        # Snap only if within the specified snap distance
        if distance_to_nearest <= snap_distance:
            snapped_geom = nearest_geom
            # Append the snapped point and original ID
            snapped_points.append({
              'geometry': snapped_geom,
              'FID' : fid,
              ID_column: point[ID_column]  # Assuming the original points have an 'FID' column
            })
            fid +=1


    # Create a new GeoDataFrame with the snapped points and original IDs
    if snapped_points:  # Only create GeoDataFrame if snapped points exist
        snapped_gdf = gpd.GeoDataFrame(snapped_points, crs=points_gdf.crs)
        # Save the output snapped points to a shapefile
        snapped_gdf.to_file(output_path)
        print(f"Snapped points (within {snap_distance} units) saved to {output_path}")
    else:
        print(f"No points snapped within {snap_distance} units.")

    return snapped_points
# ===========================================================================================================================
# Repair geometry of watershed polygons 
# ===========================================================================================================================
def repair_geometry(gdf):
    from shapely.validation import make_valid
    """
    Repair invalid geometries in a GeoDataFrame
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input GeoDataFrame with potentially invalid geometries
        
    Returns:
    --------
    GeoDataFrame
        GeoDataFrame with repaired geometries
    """
    print("Checking and repairing polygon geometries...")
    
    # Check for invalid geometries
    invalid_mask = ~gdf.is_valid
    if invalid_mask.any():
        print(f"Found {invalid_mask.sum()} invalid geometries. Repairing...")
        
        # Repair invalid geometries
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].apply(
            lambda geom: make_valid(geom) if not geom.is_valid else geom
        )
        
        # Check if repair was successful
        still_invalid = ~gdf.is_valid
        if still_invalid.any():
            print(f"Warning: {still_invalid.sum()} geometries could not be fully repaired")
        else:
            print("All geometries successfully repaired")
    else:
        print("All geometries are valid")
    
    return gdf

def simplify_geometry(gdf, tolerance=1.0):
    """
    Simplify geometries to reduce complexity while preserving topology
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    tolerance : float
        Simplification tolerance in the units of the CRS
        
    Returns:
    --------
    GeoDataFrame
        GeoDataFrame with simplified geometries
    """
    print(f"Simplifying geometries with tolerance {tolerance}...")
    
    original_count = len(gdf)
    
    # Simplify geometries
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: geom.simplify(tolerance, preserve_topology=True)
    )
    
    # Remove any empty geometries that might result from simplification
    gdf = gdf[~gdf.is_empty]
    
    if len(gdf) != original_count:
        print(f"Removed {original_count - len(gdf)} empty geometries after simplification")
    
    return gdf
#============================================================================================================================
#Function to delineate watersheds from pour points
#============================================================================================================================
def delineate_watersheds_for_pour_points(d8pointer_path,
                                         Pour_points_path,
                                         output_watershed_raster_path,
                                         watershed_polygon_path,
                                         watershed_polygon_with_pour_ID_path,
                                         ID_column):
    """
    Delineate watersheds based on specified pour points using a D8 pointer raster
    and convert the resulting raster into vector polygons.

    This function utilizes the following libraries:
    - **WhiteboxTools**: For hydrological analysis, including delineating watersheds and
      converting raster data to vector polygons.
    - **Geopandas**: For handling geospatial data and file operations.

    Parameters
    ----------
    d8pointer_path : str
        Path to the D8 pointer raster file, which is used to determine the flow direction
        and delineate watersheds.
    Pour_points_path : str
        Path to the shapefile containing snapped pour points, which are the locations from
        which watersheds will be delineated.
    ID_column : str
        The name of the ID column in the pour points shapefile used to associate pour points
        with the delineated watersheds.
    output_watershed_raster_path : str
        Path where the output watershed raster will be saved.
    watershed_polygon_path : str
        Path where the output watershed polygons shapefile will be saved.
    watershed_polygon_with_pour_ID_path : str
        Path to save a version of the watershed polygon shapefile, which may contain
        additional processing or corrections.

    Returns
    -------
    None
        The function saves the delineated watershed raster and vector polygons to the specified file paths.

    Raises
    ------
    ValueError
        If the ID_column is not a string.
    Exception
        If the watershed delineation or conversion process fails.

    Notes
    -----
    - This function utilizes the WhiteboxTools library for hydrological analysis.
    - The function first delineates watersheds from the provided D8 pointer raster based on
      the specified pour points.
    - It then converts the resulting watershed raster into vector polygons.
    - Finally, the function merges the original pour point IDs into the watershed polygon data,
      allowing for easy identification of which pour point corresponds to which watershed.

    Example
    -------
    To use this function, provide the necessary file paths and ensure that the
    WhiteboxTools library is properly set up in your environment.
    """
    if not isinstance(ID_column, str):
        raise ValueError("ID_column must be a string. Provide pour point ID column name.")
    try:
        # Delineate watersheds
        wbt.watershed(d8_pntr=d8pointer_path, output=output_watershed_raster_path, pour_pts=Pour_points_path)

        # Load the pour points as  a geodataframe
        point_gdf= gpd.read_file(Pour_points_path)
        # Convert watershed raster to vector polygons
        wbt.raster_to_vector_polygons(i=output_watershed_raster_path, output=watershed_polygon_path)

        # Load and process watershed polygons
        watershed_poly_gdf = gpd.read_file(watershed_polygon_path)
        watershed_poly_gdf['VALUE'] = watershed_poly_gdf['VALUE'].astype(int)

        # Tranafering the original ID of the pour points to the watershed polygon file
        # Merge gdf1 and gdf2 on 'FID' and 'VALUE' columns
        watershed_poly_gdf_merged = pd.merge(watershed_poly_gdf, point_gdf[['FID', ID_column]], left_on='VALUE', right_on='FID', how='left')
        watershed_poly_gdf_merged['geometry'] = watershed_poly_gdf_merged['geometry'].buffer(0) 
        
        # Drop duplicate watershed polygons based on unique pour point ID
        watershed_poly_gdf_merged = watershed_poly_gdf_merged.drop_duplicates(subset=ID_column)
        # Repair and simplify geometry 
        watershed_poly_gdf_merged=repair_geometry(watershed_poly_gdf_merged)
        watershed_poly_gdf_merged=simplify_geometry(watershed_poly_gdf_merged, tolerance=1.0)
        # Save resulting watershed polygons
        watershed_poly_gdf_merged.to_file(watershed_polygon_with_pour_ID_path)


        print("Watershed delineation and conversion complete.")

    except Exception as e:
        print(f"Error: {e}")


#======================================================================================================================================================
#                                       Function to calculate TIME of CONCENTRATION for all watersheds
#=======================================================================================================================================================
# ----------------------------------------- Function to resample raster data ----------------
def resample_raster(input_raster_data, input_raster_transform, input_raster_crs, target_raster_array, target_raster):


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
        resampling=Resampling.nearest
    )
    return resampled_raster
# ----------------------------------------- Function to calcualte mean slope of watersheds ----------------
def calculate_watershed_mean_slope(dem_path, flow_accumulation_path, watersheds_path, output_path):
    """
    Calculates mean slope for each watershed.

    This function:
    1. Loads Watershed Polygons: Reads watershed vector data.
    2. Loads DEM and Flow Accumulation Rasters: Opens DEM and flow accumulation raster files.
    3. Clips and Resamples Rasters: Clips and resamples DEM and flow accumulation rasters for each watershed.
    4. Calculates Statistics: Computes mean slope for each watershed.
    5. Saves Output: Writes results to a GeoDataFrame.

    Parameters:
    - dem_path (str): Path to DEM raster file.
    - flow_accumulation_path (str): Path to flow accumulation raster file.
    - watersheds_path (str): Path to watershed vector file.
    - output_path (str): Path to save output GeoDataFrame.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing calculated statistics for each watershed.

    Libraries Used:
    - geopandas
    - rasterio
    - numpy
    """

    # Load the watershed polygons
    watersheds = gpd.read_file(watersheds_path)
    watersheds["geometry"] = watersheds["geometry"].buffer(0)
    
    # Initialize the AvgSL column with default values
    watersheds['AvgSL'] = 0.0
    
    # Open the DEM raster and the flow accumulation raster
    with rasterio.open(dem_path) as dem, rasterio.open(flow_accumulation_path) as flow_accumulation_raster:
        # Ensure CRS match
        if dem.crs != watersheds.crs:
            watersheds = watersheds.to_crs(dem.crs)

        # Loop over each watershed polygon
        for index, (_, watershed) in enumerate(watersheds.iterrows()):
            try:
                # Clip flow accumulation raster to the current watershed
                watershed_geometry = [watershed.geometry]
                flow_accumulation_clipped, _ = mask(flow_accumulation_raster, watershed_geometry, crop=True)
                flow_accumulation_clipped = flow_accumulation_clipped[0]  # Single band flow accumulation
                flow_accumulation_clipped = np.where(flow_accumulation_clipped == flow_accumulation_raster.nodata, np.nan, flow_accumulation_clipped)

                # Clip DEM and resample to match the flow accumulation raster
                dem_clipped, _ = mask(dem, watershed_geometry, crop=True)
                dem_clipped = dem_clipped[0]  # Single band DEM
                dem_resampled = resample_raster(dem_clipped, dem.transform, dem.crs, flow_accumulation_clipped, flow_accumulation_raster)

                # Remove no-data values
                dem_resampled = np.where(dem_resampled == dem.nodata, np.nan, dem_resampled)

                # Check if we have enough data points for gradient calculation
                if dem_resampled.size < 4:  # Need at least 2x2 array for gradient
                    print(f"Warning: Watershed {index} too small for slope calculation, using default value 1.0")
                    mean_slope_percentage = 1.0  # Default slope value
                else:
                    # Check array dimensions
                    if dem_resampled.shape[0] < 2 or dem_resampled.shape[1] < 2:
                        print(f"Warning: Watershed {index} dimensions too small {dem_resampled.shape}, using default value 1.0")
                        mean_slope_percentage = 1.0  # Default slope value
                    else:
                        # Calculate slope (in percentage) using central differences in x and y directions
                        try:
                            x, y = np.gradient(dem_resampled, dem.res[0], dem.res[1])  # Gradient in x and y directions
                            slope_radians = np.arctan(np.sqrt(x**2 + y**2))
                            slope_percentage = np.tan(slope_radians) * 100  # Convert slope to percentage
                            mean_slope_percentage = np.nanmean(slope_percentage)
                            
                            # Check if the result is valid
                            if np.isnan(mean_slope_percentage) or np.isinf(mean_slope_percentage):
                                print(f"Warning: Invalid slope calculation for watershed {index}, using default value 1.0")
                                mean_slope_percentage = 1.0
                                
                        except Exception as gradient_error:
                            print(f"Warning: Gradient calculation failed for watershed {index}: {gradient_error}, using default value 1.0")
                            mean_slope_percentage = 1.0  # Default slope value

                watersheds.loc[index, 'AvgSL'] = mean_slope_percentage
                
            except Exception as watershed_error:
                print(f"Warning: Error processing watershed {index}: {watershed_error}, using default slope value 1.0")
                watersheds.loc[index, 'AvgSL'] = 1.0  # Default slope value

    # Save the output to a shapefile
    watersheds.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8', mode='w')
    return watersheds
#--------------------------------- Function to calculate the longest main channel path for each watershed
def calculate_longest_channel_path(dem_file_path,
                                   watershed_file_path,
                                   save_output_file_path,
                                   temp_folder_path,
                                   WS_ID):
    """
    Calculates longest channel path for each watershed.

    This function:
    1. Loads Watershed Data: Reads watershed shapefile.
    2. Iterates Watersheds: Clips DEM raster for each watershed.
    3. Computes Longest Flowpath: Uses WhiteboxTools to calculate longest upslope flowpath length.
    4. Extracts Maximum Flowpath Length: Reads and processes raster data.
    5. Saves Output: Writes results to shapefile.

    Parameters:
    - dem_file_path (str): Path to DEM raster file.
    - watershed_file_path (str): Path to watershed shapefile.
    - save_output_file_path (str): Path to save output shapefile.
    - WS_ID (str): Watershed ID column name.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing calculated longest channel path for each watershed.

    Libraries Used:
    - geopandas
    - rasterio
    - WhiteboxTools
    - numpy
    """

    # Read the watershed shapefile
    watersheds = gpd.read_file(watershed_file_path)
    watersheds['ChLen_m'] = np.nan

    # Iterate through each watershed polygon
    for index, row in watersheds.iterrows():
        watershed_id = row[WS_ID]  # or whatever unique identifier you have

        # Clip the DEM raster with the watershed polygon
        with rasterio.open(dem_file_path) as src:
            # Clip the raster with the watershed geometry
            clipped_raster, clipped_transform = mask(src, [row.geometry], crop=True)
            file_path = os.path.join(temp_folder_path, 'junk')
            os.makedirs(file_path, exist_ok=True)
            
            # Define a temporary output file path
            temp_clipped_dem_path = os.path.join(file_path, f"clipped_dem_{watershed_id}.tif")

            # Update metadata for the new raster
            metadata = src.meta.copy()
            metadata.update({
                'height': clipped_raster.shape[1],
                'width': clipped_raster.shape[2],
                'transform': clipped_transform
            })

            # Save the clipped raster
            with rasterio.open(temp_clipped_dem_path, 'w', **metadata) as dst:
                dst.write(clipped_raster)

        # Optionally, print the path of the saved clipped raster
        print(f"Saved clipped DEM for watershed {watershed_id} to {temp_clipped_dem_path}")

        temp_upslope_length_path = os.path.join(file_path, f'upslope_length_{watershed_id}.tif')
        wbt.max_upslope_flowpath_length(dem=temp_clipped_dem_path,
                                        output=temp_upslope_length_path)

        # Read the upslope length raster using rasterio
        with rasterio.open(temp_upslope_length_path) as src:
            raster_array = src.read(1)  # Read the first band

        # Assign the maximum value to the corresponding row and column in the watershed DataFrame
        watersheds.at[index, 'ChLen_m'] = np.max(raster_array)

    # Clean up temporary files
    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        print("File not found:", file_path)

    # Save the resulting watershed data to the output shapefile
    watersheds.to_file(save_output_file_path, driver='ESRI Shapefile', encoding='utf-8', mode='w')

    return watersheds


# -----------------------------------Function to calculate the longest overaland flow path
def calculate_max_overland_flow_path_length(pour_points_path, watershed_path, output_path, WS_ID):
    """
    Calculates maximum overland flow path length for each watershed.

    This function:
    1. Loads Pour Points and Watershed Data: Reads shapefiles.
    2. Ensures Consistent CRS: Aligns pour points and watershed CRS.
    3. Initializes Max Distance Column: Creates new column.
    4. Calculates Longest Euclidean Distance: Measures distance from pour point to watershed boundary.
    5. Updates Watershed GeoDataFrame: Assigns max distance.
    6. Saves Output: Writes updated watershed shapefile.

    Parameters:
    - pour_points_path (str): Path to pour points shapefile.
    - watershed_path (str): Path to watershed shapefile.
    - output_path (str): Path to save updated watershed shapefile.
    - WS_ID (str): Watershed ID column name.

    Returns:
    - geopandas.GeoDataFrame: Updated watershed GeoDataFrame with max overland flow path length.

    Libraries Used:
    - geopandas
    - shapely
    """

    # Load shapefiles
    pour_points_gdf = gpd.read_file(pour_points_path)
    watershed_gdf = gpd.read_file(watershed_path)

    # Ensure both GeoDataFrames have the same CRS
    if pour_points_gdf.crs != watershed_gdf.crs:
        pour_points_gdf = pour_points_gdf.to_crs(watershed_gdf.crs)
    watershed_gdf = watershed_gdf.explode(index_parts=False)

    # Initialize new column for maximum distance
    watershed_gdf['OvLen_m'] = 0.0
    
    
    # Calculate the longest Euclidean distance for each pour point to its watershed boundary points
    for idx, watershed in watershed_gdf.iterrows():
        # Get the corresponding pour point geometry
        point = pour_points_gdf.loc[pour_points_gdf[WS_ID] == watershed[WS_ID], 'geometry'].values[0]

        # Get the boundary of the watershed as a MultiLineString or LineString.
        boundary = watershed.geometry.boundary

        if boundary.geom_type == 'MultiLineString':
            # If boundary is a MultiLineString, calculate max distance over all LineStrings
            max_distance = 0
            for line in boundary.geoms:
                #Sample points along the watershed boundary
                boundary_points = [Point(coord) for coord in line.coords]

                # Calculate distances from the pour point to each boundary point
                distances = [point.distance(boundary_point) for boundary_point in boundary_points]

                # Find the maximum distance and assign it to the GeoDataFrame
                max_distance = max(max_distance, max(distances))
            watershed_gdf.at[idx, 'OvLen_m'] = max_distance


        elif boundary.geom_type == 'LineString':
             # Sample points along the watershed boundary
            boundary_points = [Point(coord) for coord in boundary.coords]

            # Calculate distances from the pour point to each boundary point
            distances = [point.distance(boundary_point) for boundary_point in boundary_points]

            # Find the maximum distance and assign it to the GeoDataFrame
            watershed_gdf.at[idx, 'OvLen_m'] = max(distances)
        else:
            watershed_gdf.at[idx, 'OvLen_m'] = 0.0  # Handle other boundary types if needed

    watershed_gdf = watershed_gdf.loc[watershed_gdf.groupby(WS_ID)['OvLen_m'].idxmax()]
    watershed_gdf_merged = watershed_gdf.dissolve(by='Point_ID', as_index=False)
    # Save updated watershed with the new column
    watershed_gdf_merged.to_file(output_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')

    return watershed_gdf_merged

# -----------------------------------Function to calculate the time of concentration
def calculate_time_of_concentration(input_watershed_polygon_path,save_ouput_path):

      """
      Calculates time of concentration for watersheds.

      This function:
      1. Loads Watershed Polygon Data: Reads input shapefile.
      2. Computes Overland and Channel Flow Times: Applies Kinematic Wave equations.
      3. Calculates Total Time of Concentration: Sums overland and channel flow times.
      4. Updates Watershed GeoDataFrame: Adds time of concentration columns.
      5. Saves Output: Writes updated watershed shapefile.

      Parameters:
      - input_watershed_polygon_path (str): Path to input watershed shapefile.
      - save_output_path (str): Path to save updated watershed shapefile.

      Returns:
      - geopandas.GeoDataFrame: Updated watershed GeoDataFrame with time of concentration.

      Libraries Used:
      - geopandas
      """

      ws_gdf = gpd.read_file(input_watershed_polygon_path)
      # Calculate polygon areas
      ws_gdf['area_sqm'] = ws_gdf.area
      ws_gdf['area_ha'] = ws_gdf['area_sqm'] / 10000
      Sov = ws_gdf['AvgSL']/100
      Sch = Sov  #####################  Assuming that the overland and channel slope are same
      Nov = 0.8
      Kov = 1.44
      Kch = 0.0195
      Lov = ws_gdf['OvLen_m']
      Lch = ws_gdf['ChLen_m']
      tov = Kov * ((Lov * Nov) ** 0.467) * (Sov ** -0.235)
      tch = Kch * (Lch ** 0.770) * (Sch ** -0.385)
      Tc = tov + tch
      ws_gdf['TOVmin'] = round(tov, 2)
      ws_gdf['TCHmin'] = round(tch, 2)
      ws_gdf['TCmin'] = round(Tc, 2)
      ws_gdf['OvLen_m'] = round(ws_gdf['OvLen_m'], 2)
      ws_gdf['ChLen_m'] = round(ws_gdf['ChLen_m'], 2)
      ws_gdf['area_ha'] = round(ws_gdf['area_ha'], 2)
      ws_gdf.to_file(save_ouput_path, driver='ESRI Shapefile',encoding='utf-8', mode='w')

      return ws_gdf
#=============================================================================================================================
#Function to filter pour points and watersheds based on drainage area in Ha
#=============================================================================================================================
def filter_watersheds_by_drainage_area(reprojected_pour_points_input_file_path,
                                       watershed_polygon_with_pour_ID_path,
                                       pour_points_path,
                                       filtered_watershed_path,
                                       filtered_point_path,
                                       ID_column,
                                       min_area_ha=2):
    """
    Calculate watershed areas, filter watersheds based on a minimum area threshold,
    and update pour points accordingly with clean column naming.
    """
    # Original input pour point shapefile
    orig_pour_gdf = gpd.read_file(reprojected_pour_points_input_file_path)
    
    # Add missing coordinate columns if absent
    if "Longitude" not in orig_pour_gdf.columns:
        orig_pour_gdf['Longitude'] = orig_pour_gdf.geometry.x
    if "Latitude" not in orig_pour_gdf.columns:
        orig_pour_gdf['Latitude'] = orig_pour_gdf.geometry.y

    # Ensure WGS84 coordinates for Orig_Lon/Orig_Lat
    if orig_pour_gdf.crs and orig_pour_gdf.crs != 'EPSG:4326':
        orig_pour_wgs84 = orig_pour_gdf.to_crs('EPSG:4326')
        orig_pour_gdf['Orig_Lon'] = orig_pour_wgs84.geometry.x
        orig_pour_gdf['Orig_Lat'] = orig_pour_wgs84.geometry.y
    else:
        orig_pour_gdf['Orig_Lon'] = orig_pour_gdf['Longitude'] 
        orig_pour_gdf['Orig_Lat'] = orig_pour_gdf['Latitude']
    
    # Get only the columns we need from original pour points, excluding geometry
    orig_pour_columns = [col for col in orig_pour_gdf.columns if col != 'geometry']
    orig_pour_gdf_no_geometry = orig_pour_gdf[orig_pour_columns]
    
    # Load watershed polygon shapefile
    watershed_gdf = gpd.read_file(watershed_polygon_with_pour_ID_path)

    # Calculate polygon areas
    watershed_gdf['area_sqm'] = watershed_gdf.area
    watershed_gdf['area_ha'] = watershed_gdf['area_sqm'] / 10000

    # Filter watershed polygons by area
    filtered_watershed_gdf = watershed_gdf[watershed_gdf['area_ha'] >= min_area_ha]
    
    # Debug: Print column names before merge
    print("Watershed columns:", filtered_watershed_gdf.columns.tolist())
    print("Original pour point columns:", orig_pour_gdf_no_geometry.columns.tolist())
    
    # Merge with original pour point data, handling overlapping columns carefully
    common_columns = set(filtered_watershed_gdf.columns) & set(orig_pour_gdf_no_geometry.columns)
    common_columns.discard('Point_ID')  # Keep Point_ID for merging
    print("Common columns found:", list(common_columns))
    
    # Create a clean version of orig_pour_gdf_no_geometry with only non-conflicting columns
    # Keep Point_ID for merging and any columns that don't exist in watershed_gdf
    columns_to_keep = ['Point_ID']
    for col in orig_pour_gdf_no_geometry.columns:
        if col not in filtered_watershed_gdf.columns or col == 'Point_ID':
            columns_to_keep.append(col)
        elif col == 'FID':
            # Rename FID to avoid conflict and keep original FID info
            orig_pour_gdf_no_geometry = orig_pour_gdf_no_geometry.rename(columns={'FID': 'Orig_FID'})
            columns_to_keep.append('Orig_FID')
    
    # Remove duplicates from columns_to_keep
    columns_to_keep = list(dict.fromkeys(columns_to_keep))
    orig_pour_clean = orig_pour_gdf_no_geometry[columns_to_keep]
    
    print("Columns to keep from original pour points:", columns_to_keep)
    
    # Perform merge
    filtered_watershed_gdf_merged = pd.merge(
        filtered_watershed_gdf, 
        orig_pour_clean, 
        left_on='Point_ID', 
        right_on='Point_ID', 
        how='left'
    )
    
    print("Merged watershed columns:", filtered_watershed_gdf_merged.columns.tolist())
    
    # Filter pour points
    pour_gdf = gpd.read_file(pour_points_path)  # pour points snapped to RSCS
    polygon_ids = filtered_watershed_gdf[ID_column].unique()
    filtered_point_gdf = pour_gdf[np.in1d(pour_gdf[ID_column], polygon_ids)].copy()
    
    # Add Longitude and Latitude in current CRS (likely UTM)
    filtered_point_gdf['Longitude'] = filtered_point_gdf.geometry.x
    filtered_point_gdf['Latitude'] = filtered_point_gdf.geometry.y
    
    print("Filtered point columns before merge:", filtered_point_gdf.columns.tolist())
    
    # Handle merge for pour points, avoiding column conflicts
    # Create a clean version for pour point merge
    point_columns_to_keep = ['Point_ID']
    for col in orig_pour_clean.columns:
        if col not in filtered_point_gdf.columns or col == 'Point_ID':
            point_columns_to_keep.append(col)
        elif col == 'FID':
            # If there's still an FID conflict, rename it
            orig_pour_clean = orig_pour_clean.rename(columns={'FID': 'Orig_FID'})
            point_columns_to_keep.append('Orig_FID')
    
    # Remove duplicates
    point_columns_to_keep = list(dict.fromkeys(point_columns_to_keep))
    orig_pour_for_points = orig_pour_clean[point_columns_to_keep]
    
    print("Columns to keep for point merge:", point_columns_to_keep)
    
    filtered_point_gdf_merged = pd.merge(
        filtered_point_gdf, 
        orig_pour_for_points, 
        left_on='Point_ID', 
        right_on='Point_ID', 
        how='left'
    )
    
    print("Final point columns:", filtered_point_gdf_merged.columns.tolist())

    # Drop duplicate watershed polygons based on unique pour point ID
    filtered_watershed_gdf_merged = filtered_watershed_gdf_merged.drop_duplicates(subset='Point_ID')

    # Check for duplicate columns before saving
    watershed_cols = filtered_watershed_gdf_merged.columns.tolist()
    point_cols = filtered_point_gdf_merged.columns.tolist()
    
    if len(watershed_cols) != len(set(watershed_cols)):
        print("WARNING: Duplicate columns found in watershed data!")
        # Remove duplicates by keeping first occurrence
        filtered_watershed_gdf_merged = filtered_watershed_gdf_merged.loc[:, ~filtered_watershed_gdf_merged.columns.duplicated()]
    
    if len(point_cols) != len(set(point_cols)):
        print("WARNING: Duplicate columns found in point data!")
        # Remove duplicates by keeping first occurrence
        filtered_point_gdf_merged = filtered_point_gdf_merged.loc[:, ~filtered_point_gdf_merged.columns.duplicated()]

    # Save filtered watershed polygons and points
    filtered_watershed_gdf_merged.to_file(filtered_watershed_path)
    filtered_point_gdf_merged.to_file(filtered_point_path)
    
#=============================================================================================================================
#Function to find road an stream crossing intersections
#=============================================================================================================================
def find_intersections_of_polylines(polyline1_path, polyline2_path, output_point_path,
                                    gauging_sta_pour_path=None, ID_column=None,
                                    pour_point_snap_distance_m=None):
    """
    Finds intersection points between two polyline shapefiles and saves the resulting
    intersection points to a new point shapefile, with optional ID column assignment.
    Adds standard metadata columns with 'NA' if not present, and ensures Lat/Lon in EPSG:4326.

    If gauging_sta_pour_path is provided, each gauging station point is snapped to the
    nearest intersection point within pour_point_snap_distance_m, and if snapped, that
    intersection point's geometry and overlapping attributes are replaced by the gauging
    station point's geometry and attributes. If not snapped (farther than the threshold),
    the original intersection point remains unchanged.
    """

    # -------------------------
    # Load polylines & align CRS
    # -------------------------
    polyline1 = gpd.read_file(polyline1_path)
    polyline2 = gpd.read_file(polyline2_path)

    if polyline1.crs != polyline2.crs:
        polyline2 = polyline2.to_crs(polyline1.crs)

    # -------------------------
    # Intersections
    # -------------------------
    poly1_union = unary_union(polyline1.geometry)
    poly2_union = unary_union(polyline2.geometry)
    intersections = poly1_union.intersection(poly2_union)

    if intersections.is_empty:
        print("No intersections found.")
        return

    # Create GeoDataFrame
    if intersections.geom_type == 'Point':
        intersection_points = gpd.GeoDataFrame(geometry=[intersections], crs=polyline1.crs)
    elif intersections.geom_type == 'MultiPoint':
        intersection_points = gpd.GeoDataFrame(geometry=list(intersections.geoms), crs=polyline1.crs)
    else:
        print("Intersections found, but not of type Point or MultiPoint.")
        return

    # -------------------------
    # IDs
    # -------------------------
    if ID_column and ID_column not in intersection_points.columns:
        intersection_points[ID_column] = [float(i + 1) for i in range(len(intersection_points))]

    if 'FID' not in intersection_points.columns:
        intersection_points['FID'] = [float(i + 1) for i in range(len(intersection_points))]

    # -------------------------
    # Required columns
    # -------------------------
    required_columns = [
        'Point_Name', 'PourSha', 'Width_ft', 'Height_ft',
        'Flag_Gst', 'Grp_ID', 'Grp_Size', 'Rte_No', 'Milepost',
        'Purpose', 'Material', 'Condition', 'GWS_ID'
    ]
    for col in required_columns:
        if col not in intersection_points.columns:
            intersection_points[col] = "NA"

    # -------------------------
    # Snap GST points (if provided)
    # -------------------------
    if gauging_sta_pour_path is not None:
        if pour_point_snap_distance_m is None:
            raise ValueError("pour_point_snap_distance_m must be provided when gauging_sta_pour_path is not None")

        try:
            gdf_st = gpd.read_file(gauging_sta_pour_path)
            # Reproject both to a metric CRS for distance calc
            try:
                metric_crs = intersection_points.estimate_utm_crs()
            except Exception:
                # Fallback to Web Mercator if estimate_utm_crs is not available
                metric_crs = "EPSG:3857"

            ip_metric = intersection_points.to_crs(metric_crs)
            gst_metric = gdf_st.to_crs(metric_crs)

            # Work with copies to avoid chained assignment warnings
            ip_metric = ip_metric.copy()

            # For each GST point, snap to nearest intersection (if within distance)
            for gst_idx, gst_row in gst_metric.iterrows():
                gst_geom = gst_row.geometry
                if gst_geom is None:
                    continue

                # Compute distances to all intersection points
                dists = ip_metric.geometry.distance(gst_geom)
                nearest_idx = dists.idxmin()
                min_dist = dists.loc[nearest_idx]

                if min_dist <= pour_point_snap_distance_m:
                    # Replace geometry & overlapping attributes in *original CRS* GDF
                    # (update metric copy first)
                    ip_metric.at[nearest_idx, 'geometry'] = gst_geom

                    # Copy overlapping attributes (non-geometry)
                    common_cols = [c for c in gdf_st.columns if c in intersection_points.columns and c != 'geometry']
                    # Map gst_row values back to the original CRS frame later
                    for c in common_cols:
                        intersection_points.at[nearest_idx, c] = gdf_st.loc[gst_idx, c]

            # Bring replaced geometries back to original CRS
            # The attributes are already updated in intersection_points; we only need geometry back.
            intersection_points['geometry'] = ip_metric.to_crs(intersection_points.crs).geometry

        except Exception as e:
            print(f"Warning: Failed during snapping gauging station points: {e}")

    # -------------------------
    # Long/Lat in WGS84, plus orig copies
    # -------------------------
    if intersection_points.crs != 'EPSG:4326':
        gdf_wgs = intersection_points.to_crs(epsg=4326)
        intersection_points['Longitude'] = gdf_wgs.geometry.x
        intersection_points['Latitude'] = gdf_wgs.geometry.y
        intersection_points['Orig_Lon'] = gdf_wgs.geometry.x
        intersection_points['Orig_Lat'] = gdf_wgs.geometry.y
    else:
        intersection_points['Longitude'] = intersection_points.geometry.x
        intersection_points['Latitude'] = intersection_points.geometry.y
        intersection_points['Orig_Lon'] = intersection_points.geometry.x
        intersection_points['Orig_Lat'] = intersection_points.geometry.y

    # -------------------------
    # Column order (unchanged from your version)
    # -------------------------
    final_columns = ['FID']
    if ID_column:
        final_columns.append(ID_column)
    final_columns.extend([
        'Point_Name', 'Longitude', 'Latitude', 'Orig_Lon', 'Orig_Lat',
        'PourSha', 'Width_ft', 'Height_ft', 'Flag_Gst', 'Grp_ID', 'Grp_Size',
        'Rte_No', 'Milepost', 'Purpose', 'Material', 'Condition', 'GWS_ID',
        'geometry'
    ])

    final_columns_no_geom = [c for c in final_columns if c != 'geometry']
    ordered_cols = [c for c in final_columns_no_geom if c in intersection_points.columns]
    if 'geometry' in intersection_points.columns:
        ordered_cols.append('geometry')

    intersection_points = intersection_points[[c for c in ordered_cols if c in intersection_points.columns]]

    # Remove duplicate cols if any
    if len(intersection_points.columns) != len(set(intersection_points.columns)):
        print("WARNING: Duplicate columns detected in intersection points!")
        intersection_points = intersection_points.loc[:, ~intersection_points.columns.duplicated()]

    # -------------------------
    # Save
    # -------------------------
    intersection_points.to_file(output_point_path)
    print(f"Intersection points saved to {output_point_path}")

#====================================================================================================================================================
#Function to flag watersheds that are draining all or some portion of the area outside the specified Boundary
#====================================================================================================================================================
def flag_watersheds_with_drainage_area_outside_region_boundary(boundary_path, polygons_path, 
                                                               pour_point_path, 
                                                               save_flagged_polygon_path, 
                                                               save_flag_ID_path,
                                                               save_flag_removed_polygon_path,
                                                               save_flag_removed_pour_path,
                                                               pour_ID,
                                                               flag_area_ha=0.5):
    """
    Identifies polygons from the polygons shapefile that extend beyond a buffered boundary.

    Args:
        boundary_shp: Path to the boundary shapefile.
        polygons_shp: Path to the polygons shapefile.
        flag_distance: Distance in the CRS units to flag watersheds outside the boundary.

    Returns:
        Geopandas GeoDataFrame: A GeoDataFrame containing the overlapping polygons.
    """
    flag_distance=10
    boundary_gdf = gpd.read_file(boundary_path)
    polygons_gdf = gpd.read_file(polygons_path)
    pour_gdf = gpd.read_file(pour_point_path)
    # Ensure consistent CRS
    polygons_gdf = polygons_gdf.to_crs(boundary_gdf.crs)

    # Buffer the boundary by flag distance in meters and convert to GeoDataFrame
    buffered_boundary = gpd.GeoDataFrame(geometry=boundary_gdf.buffer(flag_distance), crs=boundary_gdf.crs)


    # Perform spatial intersection to find polygons outside the buffered boundary
    intersection = gpd.overlay(polygons_gdf, buffered_boundary, how='difference')

    # Identify overlapping polygons
    overlapping_polygons = intersection[intersection.geometry.area > 0]
    overlapping_polygons['Outside_Area_Ha']=intersection.geometry.area/10000
    overlapping_polygons=overlapping_polygons.loc[overlapping_polygons['Outside_Area_Ha']>flag_area_ha]
    flag_ID=overlapping_polygons.loc[overlapping_polygons['Outside_Area_Ha']>flag_area_ha,pour_ID]
    overlapping_polygons.to_file(save_flagged_polygon_path)
    # Sample DataFrame
    df = pd.DataFrame({
        'flag_id':flag_ID
    })
    df.to_csv(save_flag_ID_path)
    # Filter out the rows in gdf where Pour_ID is in the flag_ids
    ws_flag_removed = polygons_gdf[~polygons_gdf[pour_ID].isin(flag_ID)]
    ws_flag_removed.to_file(save_flag_removed_polygon_path)
    pour_point_flag_removed = pour_gdf[~pour_gdf[pour_ID].isin(flag_ID)]
    pour_point_flag_removed.to_file(save_flag_removed_pour_path)
    return overlapping_polygons, flag_ID


#=================================================================================================================================================
#========================*****************                       ****************************=========================================
# ******************************** FINAL FUNCTIONS FOR WATERHSED DELINEATION
#========================*****************                       ****************************=========================================
#=================================================================================================================================================


#=================================================================================================================================================
#  WATERHSED DELINEATION both culvert and gauging station data available
#=================================================================================================================================================
def watershed_delineation_point_both(work_directory_path,
                                       user_temp_dir_path,
                                       user_input_boundary_file_path,
                                       user_output_boundary_file_path,
                                       user_input_pour_points_file_path,
                                       user_reproj_output_pour_points_file_path,
                                       user_output_pour_points_file_path,
                                       user_input_dem_raster_file_path,
                                       dem_raster_temporary_file_path,
                                       user_output_dem_raster_file_path,
                                       user_output_road_file_path,
                                       user_output_pour_points_snapped_to_roads_file_path,
                                       user_output_breaklines_file_path,
                                       user_output_Road_elevated_DEM_file_path,
                                       user_output_breaklines_burned_DEM_file_path,
                                       user_output_breached_filled_DEM_file_path,
                                       user_output_D8flow_dir_file_path,
                                       user_output_D8Flow_accum_file_path,
                                       stream_raster_temporary_file_path,
                                       user_output_stream_vector_file_path,
                                       user_output_road_stream_intersect_vector_file_path,
                                       user_output_pour_points_snapped_to_RSCS_file_path,
                                       ws_raster_temporary_file_path,
                                       ws_polygon_temporary_file_path,
                                       user_output_all_ws_polygon_file_path,
                                       user_output_ws_polygon_filtered_by_area_file_path,
                                       user_output_pour_point_filtered_file_path,
                                       user_output_final_flagged_ws_polygon_filtered_by_area_file_path,
                                       user_output_save_flag_ID_path,
                                       user_output_save_flag_removed_polygon_path,
                                       user_output_save_flag_removed_pour_path,
                                       user_output_final_watershed_html_map_path,
                                       pour_ID,
                                       hydro_enforcement_select,
                                       road_fill_dem_by_m=5, 
                                       road_fill_Dem_buffer_m=2,
                                       breakline_offset_m=10,
                                       breakline_burn_Dem_by_m=10, 
                                       breakline_burn_dem_buffer_m=1,
                                       flow_accum_threshold=100,
                                       pour_point_snap_distance_m=20,
                                       filter_Watershed_min_area_ha = 2,
                                       flag_wastershed_area_outside_boundary_ha=0.5,
                                       road_data_uploaded_by_user=None,
                                       user_input_road_file_path=None,
                                       error_log_path=None,
                                       user_id=None,
                                       project_name=None,
                                       task_type='watershed_delineation',
                                       emit_progress_func=None,
                                       check_cancellation_func = None
                                       ):
    """
    Performs watershed delineation for culvert analysis with SocketIO progress tracking and cancellation.
    """
    # Setting the working directory 
    wbt.work_dir = work_directory_path
    
    # Fall back map when process cancelled    
    boundary_gdf = gpd.read_file(f"zip://{user_input_boundary_file_path}").to_crs("EPSG:4326")    
    gdf = gpd.read_file(f"zip://{user_input_pour_points_file_path}").to_crs("EPSG:4326")
    # Get the centroid of the geometry (it returns a GeoSeries)
    centroid = boundary_gdf.geometry.centroid.iloc[0]  # Take the first centroid (assuming one geometry)
    # Extract the coordinates (centroid.y is the latitude, centroid.x is the longitude)
    center = [centroid.y, centroid.x]
    folium_map = folium.Map(location=center, zoom_start=12)
    folium.GeoJson(boundary_gdf,name="Region Boundary",
                    style_function=lambda feature: {'fillColor': 'dodgerblue', 
                    'color': 'dodgerblue',
                    'weight': 2,
                    'fillOpacity': 0.2}
                    ).add_to(folium_map)
    # Create a FeatureGroup for Pour Point Data
    pour_point_layer = folium.FeatureGroup(name="Pour Point Data", show=True)
    # Add pour point data to the FeatureGroup
    for _, row in gdf.iterrows():
        coords = row.geometry.coords[0]  # Extract coordinates from the Point geometry
        folium.CircleMarker(
            location=[coords[1], coords[0]],  # Folium expects [latitude, longitude]
            radius=5,  # Adjust the radius to your liking
            color='gray',
            fill_color='white',
            fill_opacity=1,
            weight=2,
            stroke=True,
            opacity=1,
            popup=folium.Popup('Pour Point')
        ).add_to(pour_point_layer)
    
    # Add the FeatureGroup to the map
    pour_point_layer.add_to(folium_map)
    
    # Add basemap layers
    folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community",
            name="WorldTopoMap",
            control=True
        ).add_to(folium_map)
    folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
                name="WorldImagery",
                control=True
            ).add_to(folium_map)

    # Add LayerControl to toggle layers
    folium.LayerControl(collapsed=False).add_to(folium_map) 
    # Add MeasureControl
    MeasureControl(position='topleft').add_to(folium_map)
    # Add Draw control for drawing tools in the topleft
    Draw(
        export=False,  # Disable export for this instance
        position='topleft',  # Keep the drawing tools in the top-left
        draw_options={
            'polyline': {'shapeOptions': {'color': 'red'}},
            'polygon': {'shapeOptions': {'color': 'green'}},
            'circle': {'shapeOptions': {'color': 'purple'}},
            'rectangle': {'shapeOptions': {'color': 'blue'}},
            'marker': {'icon': 'glyphicon-pushpin'}
    }
    ).add_to(folium_map)
    # Add Fullscreen control
    Fullscreen().add_to(folium_map)

    # Check for cancellation at start
    check_cancellation_func(user_id, project_name, task_type)
    
                    
    #__________________________________________________________________________________________________
    # Step 1: Fetching the utm coordinates
    # _______________________________________________________________________________________________
    try:    
        My_crs = get_utm_crs_from_wgs84(user_input_boundary_file_path)
        print("Completed: UTM coordinates fetched")
        emit_progress_func(user_id, project_name, task_type, 5, "UTM coordinates fetched")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Fetching UTM coordinates: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
        
    check_cancellation_func(user_id, project_name, task_type) 
    
    # _______________________________________________________________________________________________
    # Step 2: Project boundary, culvert, and guaging station vector data to UTM coordinates 
    # _______________________________________________________________________________________________
    try:
        project_vector_data_to_utm(user_input_boundary_file_path,user_output_boundary_file_path)
        project_vector_data_to_utm(user_input_pour_points_file_path,user_reproj_output_pour_points_file_path)
        if (road_data_uploaded_by_user == 1):
            project_vector_data_to_utm(user_input_road_file_path,user_output_road_file_path)
        print("Completed: vectors projected to UTM")
        emit_progress_func(user_id, project_name, task_type, 10, "Vectors projected to UTM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in reprojecting vectors: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)  
    
    # _______________________________________________________________________________________________
    # Step 3: Project dem raster data to UTM coordinates
    # _______________________________________________________________________________________________
    try:
        reproject_raster_from_path(user_input_dem_raster_file_path,
                                    dem_raster_temporary_file_path, My_crs)
        print("Completed: rasters projected to UTM")
        emit_progress_func(user_id, project_name, task_type, 15, "Elevation raster projected to UTM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in reprojecting raster: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step4: Clip dem raster to boundary polygon
    # _______________________________________________________________________________________________
    try:
        clip_raster_with_offset(user_output_boundary_file_path,
                                dem_raster_temporary_file_path,
                                user_output_dem_raster_file_path,
                                offset_distance_m=150)
        print("Completed: rasters clipped to boundary with offset")
        emit_progress_func(user_id, project_name, task_type, 20, "Elevation raster clipped to boundary")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in clipping dem raster to boundary: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
     
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step5: Clip culvert data to boundary polygon
    # _______________________________________________________________________________________________
    try:
        clip_vector_data_to_polygon(user_reproj_output_pour_points_file_path,
                                    user_output_boundary_file_path,
                                    user_output_pour_points_file_path)
        print("Completed: vectors clipped to boundary")
        emit_progress_func(user_id, project_name, task_type, 25, "Vectors clipped to boundary") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in clipping vectors to boundary: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
      
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step6: download road data including road width from open street maps
    # _______________________________________________________________________________________________
    if road_data_uploaded_by_user != 1:
        try:
            road_gdf = download_osm_roads_with_buffer(
                user_output_boundary_file_path,
                user_output_road_file_path
            )
            print("Completed: downloaded road data and road width from OpenStreetMap")
            emit_progress_func(user_id, project_name, task_type, 30, "Downloaded road data from OpenStreetMap")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message = f"Error in downloading road data from OpenStreetMap: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        emit_progress_func(user_id, project_name, task_type, 30, "Using user uploaded road data")
        print("User uploaded road data. Skipping OpenStreetMap download.")

    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step7: snapping pour points to roads (polylines)
    # _______________________________________________________________________________________________
    try:
        snapped_points_to_polylines_gdf = snap_points_to_polyline(user_output_pour_points_file_path,
                                                                user_output_road_file_path,
                                                                user_output_pour_points_snapped_to_roads_file_path,
                                                                ID_column='Point_ID', snap_distance_m=20)
        print("Completed: snapped pour points to road layer")
        emit_progress_func(user_id, project_name, task_type, 35, "Snapped pour points to road layer")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in snapping pour points to road layer: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type) 
    
    # _______________________________________________________________________________________________
    # Step8: Forming breakline at pour point locations
    # _______________________________________________________________________________________________
    if hydro_enforcement_select=='hydroenf_required':
        try:
            breakline_segments_gdf = create_breaklines(user_output_road_file_path,
                                                    user_output_pour_points_snapped_to_roads_file_path,
                                                    user_output_breaklines_file_path, 
                                                    offset=breakline_offset_m,
                                                    road_data_uploaded_by_user=road_data_uploaded_by_user)
            print("Completed: breakline segments created")
            emit_progress_func(user_id, project_name, task_type, 40, "Breakline segments created")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in creating breakline segments: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        print("Completed: breakline segments created")
        emit_progress_func(user_id, project_name, task_type, 40, "User Skipped Hydro-enforcement-Breaklines Not Created") 
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step9: Adjusting dem to roads 
    # _______________________________________________________________________________________________
    if hydro_enforcement_select=='hydroenf_required':
        try:
            adjust_dem_along_polyline(user_output_dem_raster_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/DEM_UTM_reprojected.tif',
                                    user_output_road_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Roads_data_UTM.shp',
                                    user_output_Road_elevated_DEM_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Road_elevated_DEM_UTM_reprojected.tif', 
                                    dy=road_fill_dem_by_m, 
                                    burn=False,
                                    buffer_width=road_fill_Dem_buffer_m, 
                                    target_crs=My_crs)
            
            print(f"Completed: evelvated dem by {road_fill_dem_by_m} m along road layer")
            emit_progress_func(user_id, project_name, task_type, 45, f"Evelvated DEM by {road_fill_dem_by_m} m along road layer")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in road filling dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        print("User Skipped Hydro-enforcement-DEM Not Adjusted along Road")
        emit_progress_func(user_id, project_name, task_type, 45, "User skipped hydro-enforcement-DEM Not adjusted along road")
    
    check_cancellation_func(user_id, project_name, task_type)


    # _______________________________________________________________________________________________
    # Step10: Adjusting dem to breaklines
    # _______________________________________________________________________________________________
    if hydro_enforcement_select=='hydroenf_required':    
        try:
            adjust_dem_along_polyline(user_output_Road_elevated_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Road_elevated_DEM_UTM_reprojected.tif',
                                    user_output_breaklines_file_path,# '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_UTM.shp',
                                    user_output_breaklines_burned_DEM_file_path,#   '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif', dy=breakline_burn_Dem_by_m, burn=True,
                                    dy=breakline_burn_Dem_by_m, 
                                    burn=True,
                                    buffer_width=breakline_burn_dem_buffer_m, 
                                    target_crs=My_crs)
            
            print(f"Completed: burned dem by {breakline_burn_Dem_by_m} m along breaklines")
            emit_progress_func(user_id, project_name, task_type, 50, f"Burned DEM by {breakline_burn_Dem_by_m} m along breaklines")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in burning dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        print("User Skipped Hydro-enforcement-DEM Not Adjusted along Breaklines")
        emit_progress_func(user_id, project_name, task_type, 50, "User skipped hydro-enforcement-DEM Not adjusted along breaklines")
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step11: Breaching and Filling DEM
    # _______________________________________________________________________________________________
    if hydro_enforcement_select=='hydroenf_required':
        try:
            wbt.breach_depressions(dem=user_output_breaklines_burned_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif',
                                output=user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                                max_depth=None, max_length=None, flat_increment=None,fill_pits=True)
            print("Breached and filled dem")
            emit_progress_func(user_id, project_name, task_type, 55, "Breached and filled DEM")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in conditioning dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        try:
            wbt.breach_depressions(dem=user_output_dem_raster_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif',
                                output=user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                                max_depth=None, max_length=None, flat_increment=None,fill_pits=True)
            print("Breached and filled dem without hydroenforcement")
            emit_progress_func(user_id, project_name, task_type, 55, "Breached and filled DEM without hydro-enforcement")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in conditioning dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message    
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step12: calculating flow direction
    # _______________________________________________________________________________________________
    try:
        wbt.d8_pointer(dem = user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                        output = user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif',
                        esri_pntr=False)
        print("Completed: calculated flow direction based on d8pointer")
        emit_progress_func(user_id, project_name, task_type, 60, "Calculated flow direction based on d8pointer")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in calculating flow direction: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step13: calculating flow accumulation
    # _______________________________________________________________________________________________
    try:
        wbt.d8_flow_accumulation(i=user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif' ,
                                output=user_output_D8Flow_accum_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif',
                                out_type="cells",log=False,clip=False,pntr=True,esri_pntr=False)
        print("Completed: calculated flow accumulation")
        emit_progress_func(user_id, project_name, task_type, 65, "Generted flow accumulation raster")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in calculating flow accumulation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step14: extracting streams
    # _______________________________________________________________________________________________
    try:
        wbt.extract_streams(flow_accum=user_output_D8Flow_accum_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif',
                        output=stream_raster_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_raster_d8_DEM_UTM.tif',
                        threshold=flow_accum_threshold,zero_background=False)
        print("Completed: calculated stream raster")
        emit_progress_func(user_id, project_name, task_type, 70, "Stream raster generated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in extracting streams: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step15: converting stream raster data to vector polygon
    #_________________________________________________________________________________________________
    try:
        wbt.raster_to_vector_lines(i=stream_raster_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_raster_d8_DEM_UTM.tif',
                            output=user_output_stream_vector_file_path)#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_vector_d8_DEM_UTM.shp')
        print(f"Completed: stream vector extracted")
        emit_progress_func(user_id, project_name, task_type, 75, "Stream vector extracted")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in converting stream raster data to vector polygon: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type) 
    #_________________________________________________________________________________________________
    # Step16: Finding Road stream intersections
    #__________________________________________________________________________________________________
    try:
        find_intersections_of_polylines(user_output_road_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Roads_data_UTM.shp',
                                    user_output_stream_vector_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_vector_d8_DEM_UTM.shp',
                                    user_output_road_stream_intersect_vector_file_path,
                                    gauging_sta_pour_path=None,
                                    ID_column=pour_ID,pour_point_snap_distance_m=pour_point_snap_distance_m)
        print("Completed: Road-stream crossings identified")
        emit_progress_func(user_id, project_name, task_type, 80, "Road-stream crossings identified")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in intersecting road and stream vectors: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step17: snapping pour points to road stream crossings
    #_________________________________________________________________________________________________
    try:
        sanpped_points_to_points_gdf = snap_points_to_nearest_points_within_distance(user_output_pour_points_snapped_to_roads_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_roads_UTM.shp',
                                                                                user_output_road_stream_intersect_vector_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/road_stream_intersect_UTM.shp',
                                                                                user_output_pour_points_snapped_to_RSCS_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                                                                ID_column=pour_ID, 
                                                                                snap_distance=pour_point_snap_distance_m)
        print(f"Completed: snapped pour points to road stream crossing points within snapping distance of {pour_point_snap_distance_m}")
        emit_progress_func(user_id, project_name, task_type, 85, f"Snapped pour points within {pour_point_snap_distance_m} m distance")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in snapping points to rscs: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step18: delineate all watersheds based on pour point
    #_________________________________________________________________________________________________
    try:
        delineate_watersheds_for_pour_points(user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif',
                                        user_output_pour_points_snapped_to_RSCS_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                        ws_raster_temporary_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_raster_UTM.tif',
                                        ws_polygon_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon.shp',
                                        user_output_all_ws_polygon_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon_with_pour_ID_UTM.shp',
                                        ID_column=pour_ID)
        print("Completed: Watersheds delineated based on pour points")
        emit_progress_func(user_id, project_name, task_type,87, "Watersheds delineated") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in delineating watersheds: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step19: calculating the mean slope for all waersheds
    #_________________________________________________________________________________________________
    try:
        calculate_watershed_mean_slope(user_output_dem_raster_file_path, 
                                        user_output_D8Flow_accum_file_path, 
                                        user_output_all_ws_polygon_file_path, 
                                        user_output_all_ws_polygon_file_path)
        print('avg slope calculated')
        emit_progress_func(user_id, project_name, task_type,89, "Calculated avearge slope of watersheds") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Avg slope calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step20: calculating the mean slope for all waersheds
    #_________________________________________________________________________________________________
    try:
        calculate_longest_channel_path(user_output_breached_filled_DEM_file_path,
                                    user_output_all_ws_polygon_file_path,
                                    user_output_all_ws_polygon_file_path,
                                    user_temp_dir_path,
                                    WS_ID='Point_ID')
        print('Longest channel length calculated')
        emit_progress_func(user_id, project_name, task_type,92, "Longest channel length calculated")  
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Longest channel length calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step 21: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        calculate_max_overland_flow_path_length(user_output_pour_points_snapped_to_RSCS_file_path,
                                            user_output_all_ws_polygon_file_path,
                                            user_output_all_ws_polygon_file_path,
                                            WS_ID='Point_ID')
        print('Maximum overland flow path length estimated')
        emit_progress_func(user_id, project_name, task_type, 95, "Maximum overland flow path length estimated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Maximum overland flow path length estimation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    #_________________________________________________________________________________________________
    # Step 22: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        calculate_time_of_concentration(user_output_all_ws_polygon_file_path,
                                        user_output_all_ws_polygon_file_path)
        print('Time of concentration calculated')
        emit_progress_func(user_id, project_name, task_type, 96, "Time of concentration calculated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Time of concentration calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step 23: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        filter_watersheds_by_drainage_area(user_output_pour_points_file_path,
                                    user_output_all_ws_polygon_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon_with_pour_ID_UTM.shp',
                                    user_output_pour_points_snapped_to_RSCS_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                    #    user_output_Gauging_st_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Gauging_stations_UTM_reprojected.shp',
                                    user_output_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
                                    user_output_pour_point_filtered_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_pour_point_by_area_UTM_reprojected.shp',
                                    ID_column=pour_ID,
                                    min_area_ha=filter_Watershed_min_area_ha)
        print(f"Completed: watersheds and pour points filtered based on drianage area >={filter_Watershed_min_area_ha}")
        emit_progress_func(user_id, project_name, task_type, 97, f"Kept watersheds with area >={filter_Watershed_min_area_ha}")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in watersheds and pour points filtering: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
      
    #_________________________________________________________________________________________________
    # Step24: Flagging watersheds and their IDs that drain all or some area > user specified threshold area in ha outside the main region boundary
    #_________________________________________________________________________________________________
    try:
        outside_ws_gdf, flag_ID = flag_watersheds_with_drainage_area_outside_region_boundary(user_output_boundary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Boundary_UTM_reprojected.shp',
                                                                                        user_output_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
                                                                                        user_output_pour_point_filtered_file_path,
                                                                                        user_output_final_flagged_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_flagged_watersheds_by_area_UTM_reprojected.shp',
                                                                                        user_output_save_flag_ID_path,
                                                                                        user_output_save_flag_removed_polygon_path,
                                                                                        user_output_save_flag_removed_pour_path,
                                                                                        pour_ID=pour_ID,
                                                                                        flag_area_ha=flag_wastershed_area_outside_boundary_ha)
        print(f'Completed: Flagged watersheds with ID: {[flag_ID]} that drain all or some area > {flag_wastershed_area_outside_boundary_ha} ha outside the main region boundary')
        emit_progress_func(user_id, project_name, task_type, 98, f'Flagged watersheds that drain area located outside the AOI')
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in flagging watershed: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step25: Create and save results in an interactive html map
    # _________________________________________________________________________________________________
    try:
        map = generate_basemap_results(work_directory_path,add_layer_control=True)
        map.save(user_output_final_watershed_html_map_path)
        print("Completed: Results saved in interactive map in html")
        emit_progress_func(user_id, project_name, task_type, 5, "Results saved in interactive map in html")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in producing the final watershed map: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    task_complete_flag_path = os.path.join(user_temp_dir_path,'ws_deln_completed.json')
    with open(task_complete_flag_path, 'w') as f:
        json.dump({}, f)
    
    return map, None





#=================================================================================================================================================
#  WATERHSED DELINEATION culvert data Not available
#=================================================================================================================================================

def watershed_delineation_point_NA(work_directory_path,
                                       user_temp_dir_path,
                                       user_input_boundary_file_path,
                                       user_output_boundary_file_path,
                                       user_input_dem_raster_file_path,
                                       dem_raster_temporary_file_path,
                                       user_output_dem_raster_file_path,
                                       user_output_road_file_path,
                                       user_output_breached_filled_DEM_file_path,
                                       user_output_D8flow_dir_file_path,
                                       user_output_D8Flow_accum_file_path,
                                       stream_raster_temporary_file_path,
                                       user_output_stream_vector_file_path,
                                       user_output_road_stream_intersect_vector_file_path,
                                       ws_raster_temporary_file_path,
                                       ws_polygon_temporary_file_path,
                                       user_output_all_ws_polygon_file_path,
                                       user_output_ws_polygon_filtered_by_area_file_path,
                                       user_output_pour_point_filtered_file_path,
                                       user_output_final_flagged_ws_polygon_filtered_by_area_file_path,
                                       user_output_save_flag_ID_path,
                                       user_output_save_flag_removed_polygon_path,
                                       user_output_save_flag_removed_pour_path,
                                       user_output_final_watershed_html_map_path,
                                       pour_ID,
                                       flow_accum_threshold=100,
                                       filter_Watershed_min_area_ha = 2,
                                       flag_wastershed_area_outside_boundary_ha=0.5,
                                       pour_point_snap_distance_m=None,
                                       road_data_uploaded_by_user=None,
                                       user_input_road_file_path=None,
                                       error_log_path=None,
                                       user_id=None,
                                       project_name=None,
                                       task_type='watershed_delineation',
                                       emit_progress_func=None,
                                       check_cancellation_func = None
                                       ):
    """
    Performs watershed delineation for culvert analysis with SocketIO progress tracking and cancellation.
    """
    # Setting the working directory 
    wbt.work_dir = work_directory_path
    
    # Fall back map when process cancelled    
    boundary_gdf = gpd.read_file(f"zip://{user_input_boundary_file_path}").to_crs("EPSG:4326")    
    # Get the centroid of the geometry (it returns a GeoSeries)
    centroid = boundary_gdf.geometry.centroid.iloc[0]  # Take the first centroid (assuming one geometry)
    # Extract the coordinates (centroid.y is the latitude, centroid.x is the longitude)
    center = [centroid.y, centroid.x]
    folium_map = folium.Map(location=center, zoom_start=12)
    folium.GeoJson(boundary_gdf,name="Region Boundary",
                    style_function=lambda feature: {'fillColor': 'dodgerblue', 
                    'color': 'dodgerblue',
                    'weight': 2,
                    'fillOpacity': 0.2}
                    ).add_to(folium_map)
    
    # Add basemap layers
    folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community",
            name="WorldTopoMap",
            control=True
        ).add_to(folium_map)
    folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
                name="WorldImagery",
                control=True
            ).add_to(folium_map)

    # Add LayerControl to toggle layers
    folium.LayerControl(collapsed=False).add_to(folium_map) 
    # Add MeasureControl
    MeasureControl(position='topleft').add_to(folium_map)
    # Add Draw control for drawing tools in the topleft
    Draw(
        export=False,  # Disable export for this instance
        position='topleft',  # Keep the drawing tools in the top-left
        draw_options={
            'polyline': {'shapeOptions': {'color': 'red'}},
            'polygon': {'shapeOptions': {'color': 'green'}},
            'circle': {'shapeOptions': {'color': 'purple'}},
            'rectangle': {'shapeOptions': {'color': 'blue'}},
            'marker': {'icon': 'glyphicon-pushpin'}
    }
    ).add_to(folium_map)
    # Add Fullscreen control
    Fullscreen().add_to(folium_map)

    # Check for cancellation at start
    check_cancellation_func(user_id, project_name, task_type)
    
                    
    #__________________________________________________________________________________________________
    # Step 1: Fetching the utm coordinates
    # _______________________________________________________________________________________________
    try:    
        My_crs = get_utm_crs_from_wgs84(user_input_boundary_file_path)
        print("Completed: UTM coordinates fetched")
        emit_progress_func(user_id, project_name, task_type, 5, "UTM coordinates fetched")
    except TaskCancelledError:
        raise 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Fetching UTM coordinates: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
        
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step 2: Project vector data to UTM coordinates 
    # _______________________________________________________________________________________________
    try:
        project_vector_data_to_utm(user_input_boundary_file_path,user_output_boundary_file_path)
        if (road_data_uploaded_by_user == 1):
            project_vector_data_to_utm(user_input_road_file_path,user_output_road_file_path)
        print("Completed: vectors projected to UTM")
        emit_progress_func(user_id, project_name, task_type, 10, "Vectors projected to UTM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in reprojecting vectors: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)  
    
    # _______________________________________________________________________________________________
    # Step 3: Project dem raster data to UTM coordinates
    # _______________________________________________________________________________________________
    try:
        reproject_raster_from_path(user_input_dem_raster_file_path,
                                    dem_raster_temporary_file_path, My_crs)
        print("Completed: rasters projected to UTM")
        emit_progress_func(user_id, project_name, task_type, 15, "Elevation raster projected to UTM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in reprojecting raster: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step4: Clip dem raster to boundary polygon
    # _______________________________________________________________________________________________
    try:
        clip_raster_with_offset(user_output_boundary_file_path,
                                dem_raster_temporary_file_path,
                                user_output_dem_raster_file_path,
                                offset_distance_m=150)
        print("Completed: rasters clipped to boundary with offset")
        emit_progress_func(user_id, project_name, task_type, 20, "Elevation raster clipped to boundary")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in clipping dem raster to boundary: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
     
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step6: download road data including road width from open street maps
    # _______________________________________________________________________________________________
    if road_data_uploaded_by_user != 1:
        try:
            road_gdf = download_osm_roads_with_buffer(
                user_output_boundary_file_path,
                user_output_road_file_path
            )
            print("Completed: downloaded road data and road width from OpenStreetMap")
            emit_progress_func(user_id, project_name, task_type, 30, "Downloaded road data from OpenStreetMap")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message = f"Error in downloading road data from OpenStreetMap: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        emit_progress_func(user_id, project_name, task_type, 30, "Using user uploaded road data")
        print("User uploaded road data. Skipping OpenStreetMap download.")

    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step11: Breaching and Filling DEM
    # _______________________________________________________________________________________________
    try:
        wbt.breach_depressions(dem=user_output_dem_raster_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif',
                            output=user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                            max_depth=None, max_length=None, flat_increment=None,fill_pits=True)
        print("Breached and filled dem")
        emit_progress_func(user_id, project_name, task_type, 55, "Breached and filled DEM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in conditioning dem: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step12: calculating flow direction
    # _______________________________________________________________________________________________
    try:
        wbt.d8_pointer(dem = user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                        output = user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif',
                        esri_pntr=False)
        print("Completed: calculated flow direction based on d8pointer")
        emit_progress_func(user_id, project_name, task_type, 60, "Calculated flow direction based on d8pointer")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in calculating flow direction: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step13: calculating flow accumulation
    # _______________________________________________________________________________________________
    try:
        wbt.d8_flow_accumulation(i=user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif' ,
                                output=user_output_D8Flow_accum_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif',
                                out_type="cells",log=False,clip=False,pntr=True,esri_pntr=False)
        print("Completed: calculated flow accumulation")
        emit_progress_func(user_id, project_name, task_type, 65, "Generated flow accumulation raster")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in calculating flow accumulation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step14: extracting streams
    # _______________________________________________________________________________________________
    try:
        wbt.extract_streams(flow_accum=user_output_D8Flow_accum_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif',
                        output=stream_raster_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_raster_d8_DEM_UTM.tif',
                        threshold=flow_accum_threshold,zero_background=False)
        print("Completed: calculated stream raster")
        emit_progress_func(user_id, project_name, task_type, 70, "Stream raster generated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in extracting streams: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step15: converting stream raster data to vector polygon
    #_________________________________________________________________________________________________
    try:
        wbt.raster_to_vector_lines(i=stream_raster_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_raster_d8_DEM_UTM.tif',
                            output=user_output_stream_vector_file_path)#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_vector_d8_DEM_UTM.shp')
        print(f"Completed: stream vector extracted")
        emit_progress_func(user_id, project_name, task_type, 75, "Stream vector extracted")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in converting stream raster data to vector polygon: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step16: Finding Road stream intersections
    #__________________________________________________________________________________________________
    try:
        find_intersections_of_polylines(user_output_road_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Roads_data_UTM.shp',
                                    user_output_stream_vector_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_vector_d8_DEM_UTM.shp',
                                    user_output_road_stream_intersect_vector_file_path,
                                    gauging_sta_pour_path=None,
                                    ID_column=pour_ID,pour_point_snap_distance_m=None)
        print("Completed: Road-stream crossings identified")
        emit_progress_func(user_id, project_name, task_type, 80, "Road-stream crossings identified")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in intersecting road and stream vectors: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    #_________________________________________________________________________________________________
    # Step18: delineate all watersheds based on pour point
    #_________________________________________________________________________________________________
    try:
        delineate_watersheds_for_pour_points(user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif',
                                        user_output_road_stream_intersect_vector_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                        ws_raster_temporary_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_raster_UTM.tif',
                                        ws_polygon_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon.shp',
                                        user_output_all_ws_polygon_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon_with_pour_ID_UTM.shp',
                                        ID_column=pour_ID)
        print("Completed: Watersheds delineated based on pour points")
        emit_progress_func(user_id, project_name, task_type,87, "Watersheds delineated") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in delineating watersheds: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step19: calculating the mean slope for all waersheds
    #_________________________________________________________________________________________________
    try:
        calculate_watershed_mean_slope(user_output_dem_raster_file_path, 
                                        user_output_D8Flow_accum_file_path, 
                                        user_output_all_ws_polygon_file_path, 
                                        user_output_all_ws_polygon_file_path)
        print('avg slope calculated')
        emit_progress_func(user_id, project_name, task_type,89, "Calculated avearge slope of watersheds") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Avg slope calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step20: calculating the mean slope for all waersheds
    #_________________________________________________________________________________________________
    try:
        calculate_longest_channel_path(user_output_breached_filled_DEM_file_path,
                                    user_output_all_ws_polygon_file_path,
                                    user_output_all_ws_polygon_file_path,
                                    user_temp_dir_path,
                                    WS_ID='Point_ID')
        print('Longest channel length calculated')
        emit_progress_func(user_id, project_name, task_type,92, "Longest channel length calculated")  
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Longest channel length calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step 21: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        calculate_max_overland_flow_path_length(user_output_road_stream_intersect_vector_file_path,
                                            user_output_all_ws_polygon_file_path,
                                            user_output_all_ws_polygon_file_path,
                                            WS_ID='Point_ID')
        print('Maximum overland flow path length estimated')
        emit_progress_func(user_id, project_name, task_type, 95, "Maximum overland flow path length estimated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Maximum overland flow path length estimation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    #_________________________________________________________________________________________________
    # Step 22: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        calculate_time_of_concentration(user_output_all_ws_polygon_file_path,
                                        user_output_all_ws_polygon_file_path)
        print('Time of concentration calculated')
        emit_progress_func(user_id, project_name, task_type, 96, "Time of concentration calculated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Time of concentration calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step 23: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        filter_watersheds_by_drainage_area(user_output_road_stream_intersect_vector_file_path,
                                    user_output_all_ws_polygon_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon_with_pour_ID_UTM.shp',
                                    user_output_road_stream_intersect_vector_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                    #    user_output_Gauging_st_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Gauging_stations_UTM_reprojected.shp',
                                    user_output_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
                                    user_output_pour_point_filtered_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_pour_point_by_area_UTM_reprojected.shp',
                                    ID_column=pour_ID,
                                    min_area_ha=filter_Watershed_min_area_ha)
        print(f"Completed: watersheds and pour points filtered based on drianage area >={filter_Watershed_min_area_ha}")
        emit_progress_func(user_id, project_name, task_type, 97, f"Kept watersheds with area >={filter_Watershed_min_area_ha}")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in watersheds and pour points filtering: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
      
    #_________________________________________________________________________________________________
    # Step24: Flagging watersheds and their IDs that drain all or some area > user specified threshold area in ha outside the main region boundary
    #_________________________________________________________________________________________________
    try:
        outside_ws_gdf, flag_ID = flag_watersheds_with_drainage_area_outside_region_boundary(user_output_boundary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Boundary_UTM_reprojected.shp',
                                                                                        user_output_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
                                                                                        user_output_pour_point_filtered_file_path,
                                                                                        user_output_final_flagged_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_flagged_watersheds_by_area_UTM_reprojected.shp',
                                                                                        user_output_save_flag_ID_path,
                                                                                        user_output_save_flag_removed_polygon_path,
                                                                                        user_output_save_flag_removed_pour_path,
                                                                                        pour_ID=pour_ID,
                                                                                        flag_area_ha=flag_wastershed_area_outside_boundary_ha)
        print(f'Completed: Flagged watersheds with ID: {[flag_ID]} that drain all or some area > {flag_wastershed_area_outside_boundary_ha} ha outside the main region boundary')
        emit_progress_func(user_id, project_name, task_type, 98, f'Flagged watersheds that drain area located outside the AOI')
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in flagging watershed: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step25: Create and save results in an interactive html map
    # _________________________________________________________________________________________________
    try:
        map = generate_basemap_results(work_directory_path,add_layer_control=True)
        map.save(user_output_final_watershed_html_map_path)
        print("Completed: Results saved in interactive map in html")
        emit_progress_func(user_id, project_name, task_type, 5, "Results saved in interactive map in html")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in producing the final watershed map: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    task_complete_flag_path = os.path.join(user_temp_dir_path,'ws_deln_completed.json')
    with open(task_complete_flag_path, 'w') as f:
        json.dump({}, f)
    
    return map, None


#=================================================================================================================================================
#  WATERHSED DELINEATION culvert data Not available
#=================================================================================================================================================

def watershed_delineation_point_only_gauging_st(work_directory_path,
                                       user_temp_dir_path,
                                       user_input_boundary_file_path,
                                       user_output_boundary_file_path,
                                       user_input_pour_points_file_path,
                                       user_reproj_output_pour_points_file_path,
                                       user_output_pour_points_file_path,
                                       user_input_dem_raster_file_path,
                                       dem_raster_temporary_file_path,
                                       user_output_dem_raster_file_path,
                                       user_output_road_file_path,
                                       user_output_pour_points_snapped_to_roads_file_path,
                                       user_output_breaklines_file_path,
                                       user_output_breaklines_burned_DEM_file_path,
                                       user_output_breached_filled_DEM_file_path,
                                       user_output_D8flow_dir_file_path,
                                       user_output_D8Flow_accum_file_path,
                                       stream_raster_temporary_file_path,
                                       user_output_stream_vector_file_path,
                                       user_output_road_stream_intersect_vector_file_path,
                                       ws_raster_temporary_file_path,
                                       ws_polygon_temporary_file_path,
                                       user_output_all_ws_polygon_file_path,
                                       user_output_ws_polygon_filtered_by_area_file_path,
                                       user_output_pour_point_filtered_file_path,
                                       user_output_final_flagged_ws_polygon_filtered_by_area_file_path,
                                       user_output_save_flag_ID_path,
                                       user_output_save_flag_removed_polygon_path,
                                       user_output_save_flag_removed_pour_path,
                                       user_output_final_watershed_html_map_path,
                                       pour_ID,
                                       hydro_enforcement_select,
                                       breakline_offset_m=10,
                                       breakline_burn_Dem_by_m=10, 
                                       breakline_burn_dem_buffer_m=1,
                                       flow_accum_threshold=100,
                                       pour_point_snap_distance_m=20,
                                       filter_Watershed_min_area_ha = 2,
                                       flag_wastershed_area_outside_boundary_ha=0.5,
                                       road_data_uploaded_by_user=None,
                                       user_input_road_file_path=None,
                                       error_log_path=None,
                                       user_id=None,
                                       project_name=None,
                                       task_type='watershed_delineation',
                                       emit_progress_func=None,
                                       check_cancellation_func = None
                                       ):
    """
    Performs watershed delineation for culvert analysis with SocketIO progress tracking and cancellation.
    """
    
    # Setting the working directory 
    wbt.work_dir = work_directory_path
    
    # Fall back map when process cancelled    
    boundary_gdf = gpd.read_file(f"zip://{user_input_boundary_file_path}").to_crs("EPSG:4326")    
    # Get the centroid of the geometry (it returns a GeoSeries)
    centroid = boundary_gdf.geometry.centroid.iloc[0]  # Take the first centroid (assuming one geometry)
    # Extract the coordinates (centroid.y is the latitude, centroid.x is the longitude)
    center = [centroid.y, centroid.x]
    folium_map = folium.Map(location=center, zoom_start=12)
    folium.GeoJson(boundary_gdf,name="Region Boundary",
                    style_function=lambda feature: {'fillColor': 'dodgerblue', 
                    'color': 'dodgerblue',
                    'weight': 2,
                    'fillOpacity': 0.2}
                    ).add_to(folium_map)
    
    # Add basemap layers
    folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community",
            name="WorldTopoMap",
            control=True
        ).add_to(folium_map)
    folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
                name="WorldImagery",
                control=True
            ).add_to(folium_map)

    # Add LayerControl to toggle layers
    folium.LayerControl(collapsed=False).add_to(folium_map) 
    # Add MeasureControl
    MeasureControl(position='topleft').add_to(folium_map)
    # Add Draw control for drawing tools in the topleft
    Draw(
        export=False,  # Disable export for this instance
        position='topleft',  # Keep the drawing tools in the top-left
        draw_options={
            'polyline': {'shapeOptions': {'color': 'red'}},
            'polygon': {'shapeOptions': {'color': 'green'}},
            'circle': {'shapeOptions': {'color': 'purple'}},
            'rectangle': {'shapeOptions': {'color': 'blue'}},
            'marker': {'icon': 'glyphicon-pushpin'}
    }
    ).add_to(folium_map)
    # Add Fullscreen control
    Fullscreen().add_to(folium_map)

    # Check for cancellation at start
    check_cancellation_func(user_id, project_name, task_type)
                 
    #__________________________________________________________________________________________________
    # Step 1: Fetching the utm coordinates
    # _______________________________________________________________________________________________
    try:    
        My_crs = get_utm_crs_from_wgs84(user_input_boundary_file_path)
        print("Completed: UTM coordinates fetched")
        emit_progress_func(user_id, project_name, task_type, 5, "UTM coordinates fetched")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Fetching UTM coordinates: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
        
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step 2: Project vector data to UTM coordinates 
    # _______________________________________________________________________________________________
    try:
        project_vector_data_to_utm(user_input_boundary_file_path,user_output_boundary_file_path)
        project_vector_data_to_utm(user_input_pour_points_file_path,user_reproj_output_pour_points_file_path)
        if (road_data_uploaded_by_user == 1):
            project_vector_data_to_utm(user_input_road_file_path,user_output_road_file_path)
        print("Completed: vectors projected to UTM")
        emit_progress_func(user_id, project_name, task_type, 10, "Vectors projected to UTM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in reprojecting vectors: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type) 
    
    # _______________________________________________________________________________________________
    # Step 3: Project dem raster data to UTM coordinates
    # _______________________________________________________________________________________________
    try:
        reproject_raster_from_path(user_input_dem_raster_file_path,
                                    dem_raster_temporary_file_path, My_crs)
        print("Completed: rasters projected to UTM")
        emit_progress_func(user_id, project_name, task_type, 15, "Elevation raster projected to UTM")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in reprojecting raster: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step4: Clip dem raster to boundary polygon
    # _______________________________________________________________________________________________
    try:
        clip_raster_with_offset(user_output_boundary_file_path,
                                dem_raster_temporary_file_path,
                                user_output_dem_raster_file_path,
                                offset_distance_m=150)
        print("Completed: rasters clipped to boundary with offset")
        emit_progress_func(user_id, project_name, task_type, 20, "Elevation raster clipped to boundary")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in clipping dem raster to boundary: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
     
    check_cancellation_func(user_id, project_name, task_type)
    
    
    # _______________________________________________________________________________________________
    # Step5: Clip culvert data to boundary polygon
    # _______________________________________________________________________________________________
    try:
        clip_vector_data_to_polygon(user_reproj_output_pour_points_file_path,
                                    user_output_boundary_file_path,
                                    user_output_pour_points_file_path)
        print("Completed: vectors clipped to boundary")
        emit_progress_func(user_id, project_name, task_type, 25, "Vectors clipped to boundary") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in clipping vectors to boundary: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
      
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step6: download road data including road width from open street maps
    # _______________________________________________________________________________________________
    if road_data_uploaded_by_user != 1:
        try:
            road_gdf = download_osm_roads_with_buffer(
                user_output_boundary_file_path,
                user_output_road_file_path
            )
            print("Completed: downloaded road data and road width from OpenStreetMap")
            emit_progress_func(user_id, project_name, task_type, 30, "Downloaded road data from OpenStreetMap")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message = f"Error in downloading road data from OpenStreetMap: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        emit_progress_func(user_id, project_name, task_type, 30, "Using user uploaded road data")
        print("User uploaded road data. Skipping OpenStreetMap download.")

    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step7: snapping pour points to roads (polylines)
    # _______________________________________________________________________________________________
    try:
        snapped_points_to_polylines_gdf = snap_points_to_polyline(user_output_pour_points_file_path,
                                                                user_output_road_file_path,
                                                                user_output_pour_points_snapped_to_roads_file_path,
                                                                ID_column='Point_ID', snap_distance_m=20)
        print("Completed: snapped pour points to road layer")
        emit_progress_func(user_id, project_name, task_type, 35, "Snapped pour points to road layer")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in snapping pour points to road layer: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step8: Forming breakline at pour point locations
    # _______________________________________________________________________________________________
    
    if hydro_enforcement_select=='hydroenf_required':
        try:
            breakline_segments_gdf = create_breaklines(user_output_road_file_path,
                                                    user_output_pour_points_snapped_to_roads_file_path,
                                                    user_output_breaklines_file_path, 
                                                    offset=breakline_offset_m,
                                                    road_data_uploaded_by_user=road_data_uploaded_by_user)
            print("Completed: breakline segments created")
            emit_progress_func(user_id, project_name, task_type, 40, "Breakline segments created")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in creating breakline segments: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        print("Completed: breakline segments created")
        emit_progress_func(user_id, project_name, task_type, 40, "User Skipped Hydro-enforcement-Breaklines Not Created")    
    check_cancellation_func(user_id, project_name, task_type)
    
    # _______________________________________________________________________________________________
    # Step9: Adjusting dem to breaklines
    # _______________________________________________________________________________________________
    if hydro_enforcement_select=='hydroenf_required':
        try:
            adjust_dem_along_polyline(user_output_dem_raster_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Road_elevated_DEM_UTM_reprojected.tif',
                                    user_output_breaklines_file_path,# '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_UTM.shp',
                                    user_output_breaklines_burned_DEM_file_path,#   '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif', dy=breakline_burn_Dem_by_m, burn=True,
                                    dy=breakline_burn_Dem_by_m, 
                                    burn=True,
                                    buffer_width=breakline_burn_dem_buffer_m, 
                                    target_crs=My_crs)
            
            print(f"Completed: burned dem by {breakline_burn_Dem_by_m} m along breaklines")
            emit_progress_func(user_id, project_name, task_type, 50, f"Burned DEM by {breakline_burn_Dem_by_m} m along breaklines")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in burning dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        print("User Skipped Hydro-enforcement-DEM Not Adjusted")
        emit_progress_func(user_id, project_name, task_type, 50, "User Skipped Hydro-enforcement-DEM Not Adjusted")
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step10: Breaching and Filling DEM
    # _______________________________________________________________________________________________
    if hydro_enforcement_select=='hydroenf_required':
        try:
            wbt.breach_depressions(dem=user_output_breaklines_burned_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif',
                                output=user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                                max_depth=None, max_length=None, flat_increment=None,fill_pits=True)
            print("Breached and filled dem")
            emit_progress_func(user_id, project_name, task_type, 55, "Breached and filled DEM")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in conditioning dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message
    else:
        try:
            wbt.breach_depressions(dem=user_output_dem_raster_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breaklines_burned_DEM_UTM.tif',
                                output=user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                                max_depth=None, max_length=None, flat_increment=None,fill_pits=True)
            print("Breached and filled dem without hydroenforcement")
            emit_progress_func(user_id, project_name, task_type, 55, "Breached and filled DEM without hydro-enforcement")
        except TaskCancelledError:
            raise
        except Exception as e:
            error_message=f"Error in conditioning dem: {str(e)}"
            print(error_message)
            if error_log_path:
                with open(error_log_path, 'w') as f:
                    json.dump({"error": error_message}, f)
            return folium_map, error_message    
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step11: calculating flow direction
    # _______________________________________________________________________________________________
    try:
        wbt.d8_pointer(dem = user_output_breached_filled_DEM_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/breached_filled_DEM_UTM.tif',
                        output = user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif',
                        esri_pntr=False)
        print("Completed: calculated flow direction based on d8pointer")
        emit_progress_func(user_id, project_name, task_type, 60, "Calculated flow direction based on d8pointer")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in calculating flow direction: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step12: calculating flow accumulation
    # _______________________________________________________________________________________________
    try:
        wbt.d8_flow_accumulation(i=user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif' ,
                                output=user_output_D8Flow_accum_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif',
                                out_type="cells",log=False,clip=False,pntr=True,esri_pntr=False)
        print("Completed: calculated flow accumulation")
        emit_progress_func(user_id, project_name, task_type, 65, "Generted flow accumulation raster")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in calculating flow accumulation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    # _______________________________________________________________________________________________
    # Step13: extracting streams
    # _______________________________________________________________________________________________
    try:
        wbt.extract_streams(flow_accum=user_output_D8Flow_accum_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_accum_d8_DEM_UTM.tif',
                        output=stream_raster_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_raster_d8_DEM_UTM.tif',
                        threshold=flow_accum_threshold,zero_background=False)
        print("Completed: calculated stream raster")
        emit_progress_func(user_id, project_name, task_type, 70, "Stream raster generated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in extracting streams: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step14: converting stream raster data to vector polygon
    #_________________________________________________________________________________________________
    try:
        wbt.raster_to_vector_lines(i=stream_raster_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_raster_d8_DEM_UTM.tif',
                            output=user_output_stream_vector_file_path)#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_vector_d8_DEM_UTM.shp')
        print(f"Completed: stream vector extracted")
        emit_progress_func(user_id, project_name, task_type, 75, "Stream vector extracted")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in converting stream raster data to vector polygon: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step15: Finding Road stream intersections and snapping the gauging points to those intersected points
    #__________________________________________________________________________________________________
    try:
        find_intersections_of_polylines(user_output_road_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Roads_data_UTM.shp',
                                    user_output_stream_vector_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Stream_vector_d8_DEM_UTM.shp',
                                    user_output_road_stream_intersect_vector_file_path,
                                    gauging_sta_pour_path=user_output_pour_points_file_path,
                                    ID_column=pour_ID,
                                    pour_point_snap_distance_m=pour_point_snap_distance_m)
        print("Completed: Road-stream crossings identified")
        emit_progress_func(user_id, project_name, task_type, 80, "Road-stream crossings identified")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in intersecting road and stream vectors: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    #_________________________________________________________________________________________________
    # Step17: delineate all watersheds based on pour point
    #_________________________________________________________________________________________________
    try:
        delineate_watersheds_for_pour_points(user_output_D8flow_dir_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Flow_dir_d8pointer_DEM_UTM.tif',
                                        user_output_road_stream_intersect_vector_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                        ws_raster_temporary_file_path,#  '/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_raster_UTM.tif',
                                        ws_polygon_temporary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon.shp',
                                        user_output_all_ws_polygon_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon_with_pour_ID_UTM.shp',
                                        ID_column=pour_ID)
        print("Completed: Watersheds delineated based on pour points")
        emit_progress_func(user_id, project_name, task_type,87, "Watersheds delineated") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in delineating watersheds: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step19: calculating the mean slope for all waersheds
    #_________________________________________________________________________________________________
    try:
        calculate_watershed_mean_slope(user_output_dem_raster_file_path, 
                                        user_output_D8Flow_accum_file_path, 
                                        user_output_all_ws_polygon_file_path, 
                                        user_output_all_ws_polygon_file_path)
        print('avg slope calculated')
        emit_progress_func(user_id, project_name, task_type,89, "Calculated avearge slope of watersheds") 
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Avg slope calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step20: calculating the mean slope for all waersheds
    #_________________________________________________________________________________________________
    try:
        calculate_longest_channel_path(user_output_breached_filled_DEM_file_path,
                                    user_output_all_ws_polygon_file_path,
                                    user_output_all_ws_polygon_file_path,
                                    user_temp_dir_path,
                                    WS_ID='Point_ID')
        print('Longest channel length calculated')
        emit_progress_func(user_id, project_name, task_type,92, "Longest channel length calculated")  
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Longest channel length calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step 21: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        calculate_max_overland_flow_path_length(user_output_road_stream_intersect_vector_file_path,
                                            user_output_all_ws_polygon_file_path,
                                            user_output_all_ws_polygon_file_path,
                                            WS_ID='Point_ID')
        print('Maximum overland flow path length estimated')
        emit_progress_func(user_id, project_name, task_type, 95, "Maximum overland flow path length estimated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Maximum overland flow path length estimation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    
    #_________________________________________________________________________________________________
    # Step 22: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        calculate_time_of_concentration(user_output_all_ws_polygon_file_path,
                                        user_output_all_ws_polygon_file_path)
        print('Time of concentration calculated')
        emit_progress_func(user_id, project_name, task_type, 96, "Time of concentration calculated")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in Time of concentration calculation: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step 23: selecting watersheds and pour points based on drainage area of watersheds
    #_________________________________________________________________________________________________
    try:
        filter_watersheds_by_drainage_area(user_output_road_stream_intersect_vector_file_path,
                                    user_output_all_ws_polygon_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/all_watersheds_polygon_with_pour_ID_UTM.shp',
                                    user_output_road_stream_intersect_vector_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/pour_points_snapped_to_rscs_UTM.shp',
                                    user_output_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
                                    user_output_pour_point_filtered_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_pour_point_by_area_UTM_reprojected.shp',
                                    ID_column=pour_ID,
                                    min_area_ha=filter_Watershed_min_area_ha)
        print(f"Completed: watersheds and pour points filtered based on drianage area >={filter_Watershed_min_area_ha}")
        emit_progress_func(user_id, project_name, task_type, 97, f"Kept watersheds with area >={filter_Watershed_min_area_ha}")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in watersheds and pour points filtering: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
      
    #_________________________________________________________________________________________________
    # Step24: Flagging watersheds and their IDs that drain all or some area > user specified threshold area in ha outside the main region boundary
    #_________________________________________________________________________________________________
    try:
        outside_ws_gdf, flag_ID = flag_watersheds_with_drainage_area_outside_region_boundary(user_output_boundary_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/Boundary_UTM_reprojected.shp',
                                                                                        user_output_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_watersheds_by_area_UTM_reprojected.shp',
                                                                                        user_output_pour_point_filtered_file_path,
                                                                                        user_output_final_flagged_ws_polygon_filtered_by_area_file_path,#'/content/drive/MyDrive/Santee_WS_deli/WS_delin_outputs/filtered_flagged_watersheds_by_area_UTM_reprojected.shp',
                                                                                        user_output_save_flag_ID_path,
                                                                                        user_output_save_flag_removed_polygon_path,
                                                                                        user_output_save_flag_removed_pour_path,
                                                                                        pour_ID=pour_ID,
                                                                                        flag_area_ha=flag_wastershed_area_outside_boundary_ha)
        print(f'Completed: Flagged watersheds with ID: {[flag_ID]} that drain all or some area > {flag_wastershed_area_outside_boundary_ha} ha outside the main region boundary')
        emit_progress_func(user_id, project_name, task_type, 98, f'Flagged watersheds that drain area located outside the AOI')
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in flagging watershed: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    check_cancellation_func(user_id, project_name, task_type)
    #_________________________________________________________________________________________________
    # Step25: Create and save results in an interactive html map
    # _________________________________________________________________________________________________
    try:
        map = generate_basemap_results(work_directory_path,add_layer_control=True)
        map.save(user_output_final_watershed_html_map_path)
        print("Completed: Results saved in interactive map in html")
        emit_progress_func(user_id, project_name, task_type, 5, "Results saved in interactive map in html")
    except TaskCancelledError:
        raise
    except Exception as e:
        error_message=f"Error in producing the final watershed map: {str(e)}"
        print(error_message)
        if error_log_path:
            with open(error_log_path, 'w') as f:
                json.dump({"error": error_message}, f)
        return folium_map, error_message
    
    task_complete_flag_path = os.path.join(user_temp_dir_path,'ws_deln_completed.json')
    with open(task_complete_flag_path, 'w') as f:
        json.dump({}, f)
    
    return map, None