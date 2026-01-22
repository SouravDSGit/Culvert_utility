# utils/save_form_data.py
import glob
import os

def get_saved_ws_den_form_data(user_id, project_name, user_dir_inputs, user_dir_outputs):
    """Get saved form data and file upload states"""
    saved_data = {}
    
    try:
        # Check for uploaded files and get their names
        boundary_files = glob.glob(os.path.join(user_dir_inputs, 'boundary.*')) + \
                        glob.glob(os.path.join(user_dir_inputs, '*boundary*.*'))
        if boundary_files:
            saved_data['boundary_uploaded'] = True
            saved_data['boundary_filename'] = os.path.basename(boundary_files[0])
        
        dem_files = glob.glob(os.path.join(user_dir_inputs, '*DEM*.tif')) + \
                   glob.glob(os.path.join(user_dir_inputs, '*dem*.tif'))
        if dem_files:
            saved_data['dem_uploaded'] = True
            saved_data['dem_filename'] = os.path.basename(dem_files[0])
        
        road_files = glob.glob(os.path.join(user_dir_inputs, '*road*.*')) + \
                    glob.glob(os.path.join(user_dir_inputs, '*Road*.*'))
        if road_files:
            saved_data['road_uploaded'] = True
            saved_data['road_filename'] = os.path.basename(road_files[0])
        
        pour_files = glob.glob(os.path.join(user_dir_inputs, '*pour*.*')) + \
                    glob.glob(os.path.join(user_dir_inputs, '*Pour*.*')) + \
                    glob.glob(os.path.join(user_dir_inputs, '*culvert*.*'))
        if pour_files:
            saved_data['pour_uploaded'] = True
            saved_data['pour_filename'] = os.path.basename(pour_files[0])
        
        # Load saved form parameters from the text file created by ws_deln_results
        form_data_file = os.path.join(user_dir_outputs, 'WS_deln', 'user_ws_deln_responses.txt')
        if os.path.exists(form_data_file):
            with open(form_data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert numeric values
                        if key in ['roadFillDemByM', 'roadFillDemBufferM', 'breaklineOffsetM',
                                  'breaklineBurnDemByM', 'breaklineBurnDemBufferM', 'flowAccumThreshold',
                                  'pourPointSnapDistanceM', 'filterWatershedMinAreaHa',
                                  'flagWatershedAreaOutsideBoundaryHa']:
                            try:
                                saved_data[key] = float(value) if '.' in value else int(value)
                            except ValueError:
                                saved_data[key] = value
                        else:
                            saved_data[key] = value
                            
    except Exception as e:
        print(f"Error loading saved form data: {e}")
    
    return saved_data