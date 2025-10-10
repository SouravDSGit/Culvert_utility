# ================================================================================
# USGS DEM Utility Functions - Fresh implementation using 3DEP VRT approach
# ================================================================================
import os
import shutil
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import from_bounds
import rasterio.windows
import math


class USGSDEMFetcher:
    """
    Utility class for fetching USGS 3DEP DEM data using direct VRT access
    """
    
    def __init__(self, logger):
        self.logger = logger
        
        # Direct VRT links to USGS 3DEP data
        self.vrt_links = {
            10: "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt",
            30: "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/USGS_Seamless_DEM_1.vrt",
            60: "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/2/TIFF/USGS_Seamless_DEM_2.vrt",
        }
        
        # Try resolutions in order of preference (highest to lowest)
        self.resolution_priority = [10, 30, 60]
    
    def get_boundary_info(self, boundary_path):
        """Load and process boundary shapefile"""
        try:
            boundary_gdf = gpd.read_file(f"zip://{boundary_path}").to_crs("EPSG:4326")
            self.logger.info("Boundary shapefile loaded successfully")
            
            # Get boundary bounds with buffer
            bounds = boundary_gdf.total_bounds
            buffer_deg = 0.005  # Approximately 500m buffer
            min_x, min_y, max_x, max_y = bounds
            min_x -= buffer_deg
            min_y -= buffer_deg  
            max_x += buffer_deg
            max_y += buffer_deg
            
            self.logger.info(f"Boundary bounds with buffer: {min_x}, {min_y}, {max_x}, {max_y}")
            
            return {
                'gdf': boundary_gdf,
                'bounds': (min_x, min_y, max_x, max_y),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error reading boundary shapefile: {str(e)}")
            return {'success': False, 'error': f"Error reading boundary shapefile: {str(e)}"}
    
    def fetch_dem_from_vrt(self, bounds, resolution, output_path):
        """Fetch DEM data directly from USGS VRT"""
        min_x, min_y, max_x, max_y = bounds
        vrt_url = self.vrt_links[resolution]
        
        try:
            self.logger.info(f"Attempting to fetch {resolution}m DEM from USGS VRT...")
            
            # Validate bounds are within US limits
            if not (-130 <= min_x <= -60 and -130 <= max_x <= -60 and 
                    20 <= min_y <= 55 and 20 <= max_y <= 55):
                raise Exception("Boundary coordinates are outside reasonable US bounds")
            
            with rasterio.Env(GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_CONNECTTIMEOUT=30):
                with rasterio.open(vrt_url) as src:
                    self.logger.info(f"VRT opened successfully for {resolution}m resolution")
                    
                    # Get VRT bounds
                    vrt_bounds = src.bounds
                    self.logger.info(f"VRT bounds: {vrt_bounds}")
                    self.logger.info(f"Request bounds: {bounds}")
                    
                    # Check intersection
                    if (max_x <= vrt_bounds.left or min_x >= vrt_bounds.right or 
                        max_y <= vrt_bounds.bottom or min_y >= vrt_bounds.top):
                        raise Exception(f"Boundary does not intersect with {resolution}m DEM coverage")
                    
                    # Constrain bounds to VRT bounds
                    clipped_bounds = (
                        max(min_x, vrt_bounds.left),
                        max(min_y, vrt_bounds.bottom), 
                        min(max_x, vrt_bounds.right),
                        min(max_y, vrt_bounds.top)
                    )
                    
                    self.logger.info(f"Clipped bounds: {clipped_bounds}")
                    
                    # Create window from bounds
                    window_raw = from_bounds(*clipped_bounds, src.transform)
                    
                    # Convert to integer window (CRITICAL for rasterio write operations)
                    window = rasterio.windows.Window(
                        col_off=max(0, int(window_raw.col_off)),
                        row_off=max(0, int(window_raw.row_off)),
                        width=max(1, int(window_raw.width)),
                        height=max(1, int(window_raw.height))
                    )
                    
                    self.logger.info(f"Integer window: col_off={window.col_off}, row_off={window.row_off}, width={window.width}, height={window.height}")
                    
                    # Validate window dimensions
                    if window.width <= 0 or window.height <= 0:
                        raise Exception(f"Invalid window dimensions: width={window.width}, height={window.height}")
                    
                    # Read data from VRT
                    try:
                        data = src.read(1, window=window)
                        self.logger.info(f"Successfully read data array of shape: {data.shape}")
                    except Exception as read_error:
                        raise Exception(f"Failed to read data from VRT: {str(read_error)}")
                    
                    # Check if we got valid data
                    if data is None or data.size == 0:
                        raise Exception("No data returned from VRT")
                    
                    # Calculate the correct transform for the integer window
                    transform = rasterio.windows.transform(window, src.transform)
                    
                    # Create output profile
                    profile = {
                        'driver': 'GTiff',
                        'height': window.height,
                        'width': window.width,
                        'count': 1,
                        'dtype': data.dtype,
                        'crs': src.crs,
                        'transform': transform,
                        'compress': 'lzw',
                        'tiled': True,
                        'blockxsize': min(512, window.width),
                        'blockysize': min(512, window.height),
                        'nodata': src.nodata
                    }
                    
                    # Write the clipped DEM
                    try:
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            dst.write(data, 1)
                        self.logger.info(f"Successfully wrote DEM to {output_path}")
                    except Exception as write_error:
                        raise Exception(f"Failed to write DEM file: {str(write_error)}")
                    
                    # Verify the written file
                    if not os.path.exists(output_path):
                        raise Exception("DEM file was not created successfully")
                    
                    file_size = os.path.getsize(output_path)
                    if file_size == 0:
                        raise Exception("DEM file is empty")
                    
                    self.logger.info(f"Successfully fetched {resolution}m DEM. File size: {file_size} bytes")
                    return True
                    
        except Exception as e:
            self.logger.warning(f"Failed to fetch {resolution}m DEM: {str(e)}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False
    
    def validate_dem_coverage(self, dem_path, boundary_gdf, coverage_threshold=0.8):
        """Validate DEM coverage against boundary"""
        try:
            with rasterio.open(dem_path) as dem:
                # Get the bounds of the DEM
                dem_bounds = box(*dem.bounds)
                dem_gdf = gpd.GeoDataFrame({'geometry': [dem_bounds]}, crs=dem.crs)
                
                # Ensure CRS compatibility
                boundary_gdf_proj = boundary_gdf.to_crs(dem.crs)
                
                # Check coverage
                boundary_union = boundary_gdf_proj.unary_union
                dem_union = dem_gdf.unary_union
                
                intersection = boundary_union.intersection(dem_union)
                boundary_area = boundary_union.area
                intersection_area = intersection.area
                
                coverage_ratio = intersection_area / boundary_area
                
                self.logger.info(f"DEM coverage: {coverage_ratio:.1%} of boundary area")
                
                if coverage_ratio < coverage_threshold:
                    return {
                        'success': False,
                        'coverage_ratio': coverage_ratio,
                        'error': f'The DEM covers only {coverage_ratio:.1%} of the boundary area. Please try a smaller boundary or upload your own DEM file.'
                    }
                
                return {
                    'success': True,
                    'coverage_ratio': coverage_ratio
                }
                    
        except Exception as e:
            self.logger.error(f"Error validating DEM coverage: {str(e)}")
            return {'success': False, 'error': f"Error validating DEM coverage: {str(e)}"}
    
    def calculate_dem_stats(self, dem_path):
        """Calculate DEM statistics"""
        try:
            with rasterio.open(dem_path) as dem:
                dem_data = dem.read(1, masked=True)
                return {
                    'min_elevation': float(dem_data.min()),
                    'max_elevation': float(dem_data.max()),
                    'mean_elevation': float(dem_data.mean()),
                    'resolution': dem.res[0],
                    'crs': str(dem.crs),
                    'width': dem.width,
                    'height': dem.height
                }
        except Exception as e:
            self.logger.warning(f"Could not calculate DEM statistics: {str(e)}")
            return {}
    
    def fetch_and_process_dem(self, boundary_path, user_dir_inputs, user_dir_temp):
        """Main method to fetch and process DEM data"""
        try:
            # Load boundary shapefile
            boundary_info = self.get_boundary_info(boundary_path)
            if not boundary_info['success']:
                return {
                    'success': False,
                    'error': boundary_info['error'],
                    'status_code': 500
                }
            
            boundary_gdf = boundary_info['gdf']
            bounds = boundary_info['bounds']
            
            # Try different resolutions in order of preference
            dem_path = None
            resolution_used = None
            
            for resolution in self.resolution_priority:
                temp_dem_path = os.path.join(user_dir_temp, f'usgs_dem_{resolution}m.tif')
                
                if self.fetch_dem_from_vrt(bounds, resolution, temp_dem_path):
                    dem_path = temp_dem_path
                    resolution_used = resolution
                    break
            
            if not dem_path:
                return {
                    'success': False,
                    'error': "Failed to fetch DEM data from USGS 3DEP service. The area might be outside the US or the service is temporarily unavailable. Please try uploading your own DEM file.",
                    'status_code': 500
                }
            
            # Validate DEM coverage
            coverage_result = self.validate_dem_coverage(dem_path, boundary_gdf)
            if not coverage_result['success']:
                return {
                    'success': False,
                    'error': coverage_result['error'],
                    'status_code': 400
                }
            
            # Move to final location
            final_dem_path = os.path.join(user_dir_inputs, 'dem.tif')
            shutil.move(dem_path, final_dem_path)
            
            # Calculate statistics
            dem_stats = self.calculate_dem_stats(final_dem_path)
            
            # Clean up temp files
            try:
                if os.path.exists(user_dir_temp):
                    shutil.rmtree(user_dir_temp)
            except Exception as e:
                self.logger.warning(f"Could not clean up temp files: {str(e)}")
            
            coverage_ratio = coverage_result['coverage_ratio']
            
            return {
                'success': True,
                'message': f'<span style="font-size: 1.25rem; text-align: center;">High-resolution DEM data ({resolution_used}m) fetched from USGS 3DEP and processed successfully. Coverage: {coverage_ratio:.1%}</span>',
                'dem_stats': dem_stats,
                'coverage_percent': round(coverage_ratio * 100, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error in fetch_and_process_dem: {str(e)}")
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'status_code': 500
            }