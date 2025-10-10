#=================================================================================================
# Libraries
#=================================================================================================
# For analysis
import numpy as np
import pandas as pd
import math
import geopandas as gpd
import whitebox
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
import fiona
import shapely
from shapely.geometry import shape, Point, LineString, MultiPolygon, Polygon
from shapely.ops import transform, unary_union, nearest_points
import pyproj
from pyproj import Transformer
import osmnx as ox
import zipfile
from werkzeug.utils import secure_filename
import random
import os
import json
from flask import request, session, flash
# For Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rasterio.plot import show
from matplotlib.lines import Line2D
import folium
from folium.plugins import MeasureControl, Draw, Fullscreen, MarkerCluster
from folium.plugins import FeatureGroupSubGroup
from folium import TileLayer
import shutil, time
from osgeo import gdal
import glob
# Initializing whitebox tools
wbt = whitebox.WhiteboxTools()


# =======================================================================================================
def add_stream_layers_to_basemap(base_map, result_gdf, hillslope_polygon_path,hillslope_streams_path):
    # # Remove existing LayerControl to prevent conflicts
    # existing_layer_controls = []
    # for child_name, child in list(base_map._children.items()):
    #     if hasattr(child, '__class__') and 'LayerControl' in child.__class__.__name__:
    #         print(f"Removing existing LayerControl: {child_name}")
    #         existing_layer_controls.append(child)
    #         del base_map._children[child_name]
    gdf_stream = result_gdf.to_crs("EPSG:4326")
    print('read the streamfile')
    if os.path.exists(hillslope_polygon_path):
        gdf_hillslope = gpd.read_file(hillslope_polygon_path).to_crs("EPSG:4326")
        hillslope_gdf_stream = gpd.read_file(hillslope_streams_path).to_crs("EPSG:4326")
        print('read the hillslope')
        
        # Create a FeatureGroup for the stream network
        hillslope_group = folium.FeatureGroup(name='Hillslope', show=True)
        
        folium.GeoJson(gdf_hillslope,
                        name='Hillslope',
                        style_function=lambda feature: {
                            'color': '#8B008B',     # hisllope colour
                            'weight': 2,          # Line thickness
                            'opacity': 1        # Line opacity
                        }).add_to(hillslope_group)
        
        # Add the stream group to the base map
        hillslope_group.add_to(base_map)
        
        # Create a FeatureGroup for the stream network
        Wepp_stream_group = folium.FeatureGroup(name='Hillslope Stream Network', show=True)
        
        folium.GeoJson(hillslope_gdf_stream,
                        name='Hillslope Stream Network',
                        style_function=lambda feature: {
                            'color': '#00b3ff',     # Stream color
                            'weight': 2,          # Line thickness
                            'opacity': 1        # Line opacity
                        }).add_to(Wepp_stream_group)
        
        # Add the stream group to the base map
        Wepp_stream_group.add_to(base_map)
    # Create a FeatureGroup for the stream network
    stream_group = folium.FeatureGroup(name='SBEVA Stream Network', show=True)
        
    folium.GeoJson(gdf_stream,
                    name='SBEVA Stream Network',
                    style_function=lambda feature: {
                        'color': '#00b3ff',     # Stream color
                        'weight': 2,          # Line thickness
                        'opacity': 1        # Line opacity
                    }).add_to(stream_group)
    
    # Add the stream group to the base map
    stream_group.add_to(base_map)
     # Add LayerControl as the final step to manage all layers
    # folium.LayerControl(position='topright', collapsed=True).add_to(base_map)
    print(f"Final map has {len(base_map._children)} children including new LayerControl")
    return base_map

# =======================================================================================================
def add_hillsope_to_basemap(base_map, hillslope_path, stream_vector_path):
    gdf_hillslope = gpd.read_file(hillslope_path).to_crs("EPSG:4326")
    gdf_stream = gpd.read_file(stream_vector_path).to_crs("EPSG:4326")
    print('read the hillslope')
    
    # Create a FeatureGroup for the stream network
    hillslope_group = folium.FeatureGroup(name='Hillslope', show=True)
    
    folium.GeoJson(gdf_hillslope,
                    name='Hillslope',
                    style_function=lambda feature: {
                        'color': '#8B008B',     # hisllope colour
                        'weight': 2,          # Line thickness
                        'opacity': 1        # Line opacity
                    }).add_to(hillslope_group)
    
    # Add the stream group to the base map
    hillslope_group.add_to(base_map)
    
    # Create a FeatureGroup for the stream network
    Wepp_stream_group = folium.FeatureGroup(name='Hillslope Stream Network', show=True)
    
    folium.GeoJson(gdf_stream,
                    name='Hillslope Stream Network',
                    style_function=lambda feature: {
                        'color': '#00b3ff',     # Stream color
                        'weight': 2,          # Line thickness
                        'opacity': 1        # Line opacity
                    }).add_to(Wepp_stream_group)
    
    # Add the stream group to the base map
    Wepp_stream_group.add_to(base_map)
    
    return base_map
 
# =======================================================================================================
def extract_streamraster_from_flow_accum(input_flow_accum_path,
                                                   stream_raster_output_path,
                                                   input_d8pointer_raster_path,
                                                   stream_vector_output_path,
                                                   hillslope_polygon_path,
                                                   hillslope_streams_path,
                                                   watershed_shapefile_path,
                                                   threshold=None
                                                   ):
    
    
    wbt.extract_streams(input_flow_accum_path, 
        stream_raster_output_path, 
        threshold, 
        zero_background=False)
    
    
    wbt.raster_streams_to_vector(
        stream_raster_output_path, 
        input_d8pointer_raster_path, 
        stream_vector_output_path
        )

    boundary_crs = gpd.read_file(watershed_shapefile_path).crs
    stream_vector=gpd.read_file(stream_vector_output_path)
    result_gdf = stream_vector.set_crs(boundary_crs, inplace=False)
    result_gdf.to_file(stream_vector_output_path)

    return None

## =============================================================================
##  HILLSLOPE Delineation
##  Function to cut the hillslope polygons when road crosses them
## ================================================================================
# =======================================================================================================
def extract_streams_for_hillsope_Delin(input_flow_accum_path,
                                                   stream_raster_output_path,
                                                   input_d8pointer_raster_path,
                                                   stream_vector_output_path,
                                                   threshold=None
                                                   ):
    
    
    wbt.extract_streams(input_flow_accum_path, 
        stream_raster_output_path, 
        threshold, 
        zero_background=False)
    
    
    wbt.raster_streams_to_vector(
        stream_raster_output_path, 
        input_d8pointer_raster_path, 
        stream_vector_output_path
        )

    with rasterio.open(input_flow_accum_path) as src:
        # Get the CRS
        crs = src.crs
        
        # Print the CRS information
        print(f"CRS Authority: {crs.to_authority()}")
        print(f"CRS WKT: {crs.to_wkt()}")
        
        # If it's a UTM projection, try to determine the zone
        if crs.is_projected:
            print(f"Projected CRS: {crs.to_string()}")
            
            # Basic info about the coordinate ranges
            print(f"X range: {src.bounds.left} to {src.bounds.right}")
            print(f"Y range: {src.bounds.bottom} to {src.bounds.top}")
    streams_vector=gpd.read_file(stream_vector_output_path)
    result_gdf = streams_vector.set_crs(crs, inplace=False)
    result_gdf.to_file(stream_vector_output_path)
    print('stream vector calculated')
    return None

def cut_hillslope_polygons_based_on_road_layer(road_line_vector_path, 
                                              polygon_shapefile_path,
                                              final_polygon_shapefile_path):
    # 1. Load the data
    polygons = gpd.read_file(polygon_shapefile_path)
    roads = gpd.read_file(road_line_vector_path)
    
    # Make sure both are in the same CRS
    if polygons.crs != roads.crs:
        roads = roads.to_crs(polygons.crs)
    
    # 2. Create function to split polygons by lines with variable buffer
    def split_polygon_by_line(polygon, line, buffer_width):
        # Convert to geometry if needed
        if hasattr(polygon, 'geometry'):
            polygon = polygon.geometry
        if hasattr(line, 'geometry'):
            line = line.geometry
        
        # Create a buffer around the line based on road_width/2
        line_buffer = line.buffer(buffer_width)
        
        # Split the polygon using the line buffer
        split_polygons = polygon.difference(line_buffer)
        
        # Return the resulting geometries
        return split_polygons
    
    # Check if the road_width column exists
    if 'road_width' not in roads.columns:
        raise ValueError("The road shapefile does not contain a 'road_width' column")
    
    # 3. Apply the split for each polygon and each road
    result_geometries = []
    for _, polygon_row in polygons.iterrows():
        current_polygons = [polygon_row.geometry]
        
        for _, road_row in roads.iterrows():
            new_polygons = []
            
            # Calculate buffer width (half of road width)
            # Use a small default if road_width is 0 or None
            if road_row['road_width'] is None or road_row['road_width'] == 0:
                buffer_width = 0.0000001  # Tiny default buffer
            else:
                buffer_width = road_row['road_width'] / 2
            
            for poly in current_polygons:
                # Skip if the road doesn't intersect this polygon
                if not poly.intersects(road_row.geometry):
                    new_polygons.append(poly)
                    continue
                
                # Split the polygon and add results
                split_result = split_polygon_by_line(poly, road_row.geometry, buffer_width)
                
                # Handle both single polygons and multipolygons in the result
                if isinstance(split_result, MultiPolygon):
                    new_polygons.extend(list(split_result.geoms))
                elif isinstance(split_result, Polygon):
                    new_polygons.append(split_result)
                else:
                    # Handle empty geometries or other types if they occur
                    if not split_result.is_empty:
                        new_polygons.append(split_result)
            
            # Update current polygons with the new split polygons
            current_polygons = new_polygons
        
        # Add all resulting polygons to the final list
        result_geometries.extend(current_polygons)
    
    # 4. Create a new GeoDataFrame with the split polygons
    result_gdf = gpd.GeoDataFrame(geometry=result_geometries, crs=polygons.crs)
    
    # 5. Save the result
    result_gdf.to_file(final_polygon_shapefile_path)
    
    return None

def delineate_hillsopes_for_WEPP(boundary_shapefile_path,
                                 road_line_vector_path,
                                 input_flow_accum_path,
                                 d8_pntr_path,
                                 streams_raster_path,
                                 stream_vector_path,
                                 hillslope_raster_output_path,
                                 hillslope_polygon_output_vector_path,
                                 base_map,
                                 threshold=None,
                                 cutPolygon=None):
    
    # wbt.extract_streams(input_flow_accum_path, 
    #     streams_raster_path, 
    #     threshold=threshold, 
    #     zero_background=False)
    
    
    extract_streams_for_hillsope_Delin(input_flow_accum_path,
                                    streams_raster_path,
                                    d8_pntr_path,
                                    stream_vector_path,
                                    threshold=threshold
                                    )
    
    wbt.hillslopes(
        d8_pntr_path, 
        streams_raster_path, 
        hillslope_raster_output_path, 
        esri_pntr=False
        )
    wbt.raster_to_vector_polygons(i=hillslope_raster_output_path,
                            output=hillslope_polygon_output_vector_path)
    
    wbt.clip(hillslope_polygon_output_vector_path, 
            boundary_shapefile_path, 
            hillslope_polygon_output_vector_path
        )
    if (cutPolygon == 'Yes'):
        cut_hillslope_polygons_based_on_road_layer(road_line_vector_path, 
                                                hillslope_polygon_output_vector_path, 
                                                hillslope_polygon_output_vector_path)
    
    folium_map=add_hillsope_to_basemap(base_map, hillslope_polygon_output_vector_path,stream_vector_path)
    return folium_map

