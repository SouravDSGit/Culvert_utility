# Intellectual Resource or Information used from
# https://www.hydrocad.net/pdf/TR-55%20Manual.pdf
# https://nweb.eng.fiu.edu/FUENTES/PASTF2024OLD540WR/Additional%20Examples/Module%203/NRCS%20TR-55/NRCS-TR55%20Equation-based%20Method.pdf
# read the docs about the NOAA-Atlas14 api https://www.weather.gov/media/owp/hdsc_documents/NA14_Sec5_PFDS.pdf

    
"""
Verification Summary against TR-55 Manual:

+---------------------------+---------------+---------------------------------------------+
| Component                 | Status        | Notes                                       |
+---------------------------+---------------+---------------------------------------------+
| Initial Abstraction (Ia)  | ✅ Correct    | Uses Ia = 0.2S (TR-55 standard)             |
| Runoff Equation           | ✅ Correct    | Uses 0.2S and 0.8S (TR-55 standard)         |
| Internal Consistency      | ✅ Consistent | Both functions now use 0.2S                 |
| Fp Clipping               | ✅ Correct    | Clips WetAper to [0, 5] range               |
| Fp Interpolation          | ✅ Correct    |                                             |
| Unit Peak Discharge (qu)  | ✅ Correct    | Coefficients match TR-55 Appendix F         |
| Peak Discharge Equation   | ✅ Correct    | qp = qu x Am x Q x Fp                       |
+---------------------------+---------------+---------------------------------------------+
"""
    
import numpy as np
import math
import geopandas as gpd
import pandas as pd
import os
import requests
from app import TaskCancelledError
def get_state_from_point(lat, lon, usa_states_shapefile_path, state_abbr_column="stusps"):
    """
    Finds the US state abbreviation in which the given point (lat, lon) falls.
    
    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        usa_states_shapefile_path (str): Path to a known US states shapefile (should contain a state abbreviation column)
        state_abbr_column (str): Column name in the US states shapefile for state abbreviations (default 'stusps')
        
    Returns:
        str: US state abbreviation string where the point is located, or None if no state contains the point
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Create a point geometry from the lat/lon (Imp: the lat/lon must be in "EPSG:4326" crs)
        point = Point(lon, lat)  # Note: Point takes (x, y) which is (lon, lat)
        
        # Load the US states shapefile and convert to WGS84 (EPSG:4326)
        states_gdf = gpd.read_file(f"zip://{usa_states_shapefile_path}").to_crs("EPSG:4326")
        
        # Find the state that contains the point
        containing_state = states_gdf[states_gdf.contains(point)]
        
        if containing_state.empty:
            return None
            
        if state_abbr_column not in containing_state.columns:
            print(f"Column '{state_abbr_column}' not found in the states shapefile. Available columns: {containing_state.columns.tolist()}")
            return None
            
        # Extract the state abbreviation from the containing state
        state_abbr = containing_state[state_abbr_column].iloc[0]
        
        return state_abbr
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def calculate_Ia(CN):
    """
    Calculates initial abstraction using Curve Number
    Args:
        CN (integer): unitless Curve number
        
    Returns:
        float: Initial abstarction in inches
    """
    S = (1/0.0394)*((1000/CN)-10) # S is claculated in mm
    Ia = 0.2*S # in mm
    Ia = Ia * 0.0393701 # from mm to inches
    return Ia
    
    
def calculate_runoff(P, CN):
    """
    Calculates Runoff, R in cm from Curve Number

    Args:
        P (float): 24hr Precipitaion in inches
        CN (integer): unitless Curve number
        
    Returns:
        float: Runoff R in cm
    """
    S = (1/0.0394)*((1000/CN)-10) # S is claculated in mm
    P = P *25.400013716 # converting form inch to mm
    R = ((P - 0.2*S)**2)/(P + 0.8*S) # in mm
    R = R * 0.1 # mm to cm
    return R

def calculate_qu(Ia, P, Tc, type_of_P):
    """
    Calculate unit peak discharge (qu) based on Ia, P, Tc, and rainfall type using 
    the equation: log(qu) = C0 + C1*log(tc) + C2*(log(tc))^2 - 2.366
    
    Parameters:
    Ia (float): Initial abstraction in inches
    P (float): 24 hr Precipitation depth in inches
    Tc (float): Time of concentration (hours)
    type_of_P (str): Rainfall type ('I', 'IA', 'II', or 'III')
    
    Returns:
    float: Unit peak discharge (qu) in m3/s per cm per km2
    """
    try:
        # Input validation
        if not isinstance(Ia, (int, float)):
            raise TypeError(f"Ia must be a number, got {type(Ia).__name__}")
        
        if not isinstance(P, (int, float)):
            raise TypeError(f"P must be a number, got {type(P).__name__}")
        
        if not isinstance(Tc, (int, float)):
            raise TypeError(f"Tc must be a number, got {type(Tc).__name__}")
        
        if not isinstance(type_of_P, str):
            raise TypeError(f"type_of_P must be a string, got {type(type_of_P).__name__}")
        
        if Tc <= 0:
            raise ValueError(f"Tc must be positive, got {Tc}")
        
        if P <= 0:
            raise ValueError(f"P must be positive, got {P}")
        
        if Ia < 0:
            raise ValueError(f"Ia cannot be negative, got {Ia}")
        
        # Validate rainfall type
        valid_types = ['I', 'IA', 'II', 'III']
        if type_of_P not in valid_types:
            raise ValueError(f"type_of_P must be one of {valid_types}, got '{type_of_P}'")
        
        # Calculate Ia/P ratio
        Ia_P = Ia / P
        
        # Coefficient table for each rainfall type and Ia/P value
        coefficients = {
            'I': {
                0.10: {'C0': 2.30550, 'C1': -0.51429, 'C2': -0.11750},
                0.20: {'C0': 2.23537, 'C1': -0.50387, 'C2': -0.08929},
                0.25: {'C0': 2.18219, 'C1': -0.48488, 'C2': -0.06589},
                0.30: {'C0': 2.10624, 'C1': -0.45695, 'C2': -0.02835},
                0.35: {'C0': 2.00303, 'C1': -0.40769, 'C2': 0.01983},
                0.40: {'C0': 1.87733, 'C1': -0.32274, 'C2': 0.05754},
                0.45: {'C0': 1.76312, 'C1': -0.15644, 'C2': 0.00453},
                0.50: {'C0': 1.67889, 'C1': -0.06930, 'C2': 0.0}
            },
            'IA': {
                0.10: {'C0': 2.03250, 'C1': -0.31583, 'C2': -0.13748},
                0.20: {'C0': 1.91978, 'C1': -0.28215, 'C2': -0.07020},
                0.25: {'C0': 1.83842, 'C1': -0.25543, 'C2': -0.02597},
                0.30: {'C0': 1.72657, 'C1': -0.19826, 'C2': 0.02633},
                0.50: {'C0': 1.63417, 'C1': -0.09100, 'C2': 0.0}
            },
            'II': {
                0.10: {'C0': 2.55323, 'C1': -0.61512, 'C2': -0.16403},
                0.30: {'C0': 2.46532, 'C1': -0.62257, 'C2': -0.11657},
                0.35: {'C0': 2.41896, 'C1': -0.61594, 'C2': -0.08820},
                0.40: {'C0': 2.36409, 'C1': -0.59857, 'C2': -0.05621},
                0.45: {'C0': 2.29238, 'C1': -0.57005, 'C2': -0.02281},
                0.50: {'C0': 2.20282, 'C1': -0.51599, 'C2': -0.01259}
            },
            'III': {
                0.10: {'C0': 2.47317, 'C1': -0.51848, 'C2': -0.17083},
                0.30: {'C0': 2.39628, 'C1': -0.51202, 'C2': -0.13245},
                0.35: {'C0': 2.35477, 'C1': -0.49735, 'C2': -0.11985},
                0.45: {'C0': 2.24876, 'C1': -0.41314, 'C2': -0.11508},
                0.50: {'C0': 2.17772, 'C1': -0.36803, 'C2': -0.09525}
            }
        }
        
        # Get the available Ia/P values for the selected rainfall type
        available_ia_p = sorted(coefficients[type_of_P].keys())
        
        # Find the appropriate coefficients for the Ia/P value
        if Ia_P <= available_ia_p[0]:
            # Use the lowest available Ia/P
            coefs = coefficients[type_of_P][available_ia_p[0]]
        elif Ia_P >= available_ia_p[-1]:
            # Use the highest available Ia/P
            coefs = coefficients[type_of_P][available_ia_p[-1]]
        else:
            # Find the two nearest Ia/P values for interpolation
            for i in range(len(available_ia_p) - 1):
                if available_ia_p[i] <= Ia_P <= available_ia_p[i + 1]:
                    # Interpolate the coefficients
                    coefs1 = coefficients[type_of_P][available_ia_p[i]]
                    coefs2 = coefficients[type_of_P][available_ia_p[i + 1]]
                    weight = (Ia_P - available_ia_p[i]) / (available_ia_p[i + 1] - available_ia_p[i])
                    
                    # Linear interpolation for each coefficient
                    C0 = coefs1['C0'] + weight * (coefs2['C0'] - coefs1['C0'])
                    C1 = coefs1['C1'] + weight * (coefs2['C1'] - coefs1['C1'])
                    C2 = coefs1['C2'] + weight * (coefs2['C2'] - coefs1['C2'])
                    
                    coefs = {'C0': C0, 'C1': C1, 'C2': C2}
                    break
            else:
                # Fallback (shouldn't happen due to earlier checks)
                coefs = coefficients[type_of_P][available_ia_p[0]]
        
        # Calculate qu using the formula: log(qu) = C0 + C1*log(tc) + C2*(log(tc))^2 - 2.366
        log_tc = math.log10(Tc)
        log_qu = coefs['C0'] + coefs['C1'] * log_tc + coefs['C2'] * (log_tc ** 2) - 2.366
        qu = 10 ** log_qu
        
        return qu
        
    except Exception as e:
        # Re-raise the exception with a clear message
        raise ValueError(f"Error calculating qu: {str(e)}")
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
  try:
      response = requests.get(url)
      response.raise_for_status()

      # Extract filename from path (optional, for logging/messages)
      filename = os.path.basename(file_path)

      with open(file_path, 'wb') as f:
          f.write(response.content)
      if (verbose==True):
        print(f"Successfully downloaded {filename} to {file_path}")

  except requests.exceptions.RequestException as e:
      print(f"Error downloading {filename}: {e}")
      raise

###################################################################################################
######## Calculate peak discharge using the graphical peak discharge method
#################################################################################################    
def calculate_peak_Discharge_GPDM(ws_char_polygon_shapefile_path,
                                  usa_states_shapefile_path,
                                  rainfall_type_path,
                                  pidf_files_path, 
                                  GPDM_output_dir_path,
                                  RP_list, ts_type,
                                  evpr,
                                  user_id=None,
                                  project_name=None,
                                  task_type=None,
                                  check_cancellation_func=None):
    try:
        watersheds_gdf = gpd.read_file(ws_char_polygon_shapefile_path)
        myCRS=watersheds_gdf.crs
        print(myCRS)
        watersheds_gdf=watersheds_gdf.to_crs('EPSG:4326') 
        check_cancellation_func(user_id, project_name, task_type)
        # Merge (dissolve) the polygons based on the 'Point_ID'
        gdf = watersheds_gdf.dissolve(by='Point_ID')
        # reset the index to make the 'Point_ID' a regular column
        gdf = gdf.reset_index()   
        print("Wetland cover area extracted for each watershed")
        check_cancellation_func(user_id, project_name, task_type)
        # Calculate pond and swamp adjustment factor
        gdf['WetAper'] = ((gdf['WetAHa'] / gdf['area_ha'])) * 100
        # Clip to valid range
        gdf['WetAper'] = np.clip(gdf['WetAper'], 0, 5)
        gdf['Fp'] = np.interp(gdf['WetAper'], 
                            [0.0, 0.2, 1.0, 3.0, 5.0], 
                            [1.00, 0.97, 0.87, 0.75, 0.72])
        
        # Initialize empty DataFrames
        pidf_df = pd.DataFrame()
        q_df = pd.DataFrame()
        
        # Read rainfall type data (only needed if evpr is None)
        if evpr is None:
            rain_type_df = pd.read_csv(rainfall_type_path)
        check_cancellation_func(user_id, project_name, task_type)
        
        # Iterate over each polygon in the GeoDataFrame
        for index, row in gdf.iterrows():
            durgdf = row['TCmin']
            # dur, closest_duration_min = find_closest_duration(durgdf)
            # print(dur)
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
            
            # Initialize empty DataFrames for this iteration
            rvdf = pd.DataFrame()
            pidf = pd.DataFrame()
            temp_q_df = pd.DataFrame()
            
            # Initialize variables that may be needed regardless of evpr
            state = None
            type_of_P = None
            
            if evpr is None:
                # Original NOAA Atlas14 approach
                # Download precipitation data
                url = f'https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/fe_text.csv?lat={lat}&lon={lon}&data=depth&units=english&series={ts_type}'
                file_path = os.path.join(pidf_files_path, f'P_24hr_depth_{lon}_{lat}_{ts_type}_inch.txt')
                download_data(url, file_path, verbose=False)
                print('NOAA Atlas14 PIDF downloaded')
                
                # Get the state
                state = get_state_from_point(lat, lon, usa_states_shapefile_path, state_abbr_column="stusps")
                check_cancellation_func(user_id, project_name, task_type)
                # Get the rainfall distribution type
                if state in rain_type_df['Abbr'].values:
                    type_of_P = rain_type_df.loc[rain_type_df['Abbr'] == state, 'Distribution_Type'].iloc[0]
                else:
                    print(f"Warning: State abbreviation '{state}' not found in rainfall type CSV. Using Type II as default.")
                    type_of_P = "II"
            else:
                # Use provided evpr value - assume Type II distribution as default
                type_of_P = "II"
                print(f"Using provided EVPR value: {evpr} inches")
                
            # Calculate initial abstraction
            Curve_number = row['CN_val']
            Ia = calculate_Ia(Curve_number)
            Tc = durgdf/60 # changed from min to hr
            check_cancellation_func(user_id, project_name, task_type)
            
            if evpr is None:
                # Process each return period with NOAA data
                for ri in RP_list:
                    # Extract 24-hr precipitation values directly
                    rvd = extract_PI_from_NOAA(file_path, '24-hr', ri)
                    
                    # Store precipitation values
                    pidf[f'P{ri}yrL'] = [rvd[0]]
                    pidf[f'P{ri}yrE'] = [rvd[1]]
                    pidf[f'P{ri}yrU'] = [rvd[2]]
                    
                    # Calculate runoff
                    R_low = calculate_runoff(rvd[0], Curve_number)
                    R_est = calculate_runoff(rvd[1], Curve_number)
                    R_up = calculate_runoff(rvd[2], Curve_number)
                    
                    # Calculate unit peak discharge
                    qu_low = calculate_qu(Ia, rvd[0], Tc, type_of_P)
                    qu_est = calculate_qu(Ia, rvd[1], Tc, type_of_P)
                    qu_up = calculate_qu(Ia, rvd[2], Tc, type_of_P)
                    
                    # Calculate peak discharge
                    Qp_low = qu_low * R_low * row['area_ha'] * 0.01 * row['Fp']
                    Qp_est = qu_est * R_est * row['area_ha'] * 0.01 * row['Fp']
                    Qp_up = qu_up * R_up * row['area_ha'] * 0.01 * row['Fp']
                    
                    # Add peak discharge values to temporary DataFrame
                    temp_q_df[f'GP{ri}yrL'] = [round(Qp_low, 2)]
                    temp_q_df[f'GP{ri}yrE'] = [round(Qp_est, 2)]
                    temp_q_df[f'GP{ri}yrU'] = [round(Qp_up, 2)]
            else:
                # Use provided evpr value (single event calculation)
                # Store precipitation value (convert to cm like in original code)
                pidf[f'Pevent'] = [float(evpr) * 2.54]  # converting from inch to cm
                
                # Calculate runoff using evpr
                R_event = calculate_runoff(float(evpr), Curve_number)
                
                # Calculate unit peak discharge using evpr
                qu_event = calculate_qu(Ia, float(evpr), Tc, type_of_P)
                
                # Calculate peak discharge
                Qp_event = qu_event * R_event * row['area_ha'] * 0.01 * row['Fp']
                
                # Add peak discharge value to temporary DataFrame
                temp_q_df[f'GPevent'] = [round(Qp_event, 2)]
            
            check_cancellation_func(user_id, project_name, task_type)
            # Set indices for the DataFrames
            rvdf.index = [index]
            pidf.index = [index]
            temp_q_df.index = [index]
            
            # Concatenate with the main DataFrames
            pidf_df = pd.concat([pidf_df, pidf*2.54], axis=0) # changing precip depth from inch to cm and saving into a geodataframe
            q_df = pd.concat([q_df, temp_q_df], axis=0)
        
        check_cancellation_func(user_id, project_name, task_type)
        # Concatenate results with the original GeoDataFrame
        gdf_pidf = gdf.join(pidf_df, how='outer')
        output_pidf_shapefile_path = os.path.join(GPDM_output_dir_path, "P_depth_cm_per_watershed_UTM_reprojected.shp")
        gdf_pidf.to_file(output_pidf_shapefile_path)
        check_cancellation_func(user_id, project_name, task_type)
        gdf_peakQ = gdf.join(q_df, how='outer')
        gdf_peakQ=gdf_peakQ.to_crs(myCRS)
        output_qdf_shapefile_path = os.path.join(GPDM_output_dir_path, "GPDM_Q_m3perSec_per_watershed_UTM_reprojected.shp")
        gdf_peakQ.to_file(output_qdf_shapefile_path)
        
        if evpr is None:
            print(f"Status: GPDM Method based discharges in m3/s for {RP_list} year saved to {GPDM_output_dir_path}")
        else:
            print(f"Status: GPDM Method based discharge in m3/s for event precipitation ({evpr} inches) saved to {GPDM_output_dir_path}")
        return gdf_peakQ, gdf_pidf
    except TaskCancelledError:
        # Re-raise to let the calling function handle it
        raise
    except Exception as e:
        print(f"Error in Case 6 (Graphical Peak Discharge Method): {str(e)}")
 