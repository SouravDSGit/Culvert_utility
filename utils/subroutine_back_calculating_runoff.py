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
import pandas as pd
import numpy as np
import os

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


def Back_calculate_runoff_for_gauged_WSs(flow_file_dir, precip_file_dir, Gst_Names=None, select_top=10000):
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
        flow_file_path = os.path.join(flow_file_dir, "Inst_Streamflow",f"inst_streamflow_series_{gst}.csv")
        precip_file_path = os.path.join(precip_file_dir, "PI",f"full_precip_series_{gst}.csv")
        
        try:
            mean_cval = back_calculate_Coeff_of_Runoff(flow_file_path, precip_file_path, select_top)
            cval_data['Gst_ID'].append(gst)
            cval_data['MeanCvalue'].append(mean_cval)
        except Exception as e:
            print(f"Error processing {gst}: {e}")
            cval_data['Gst_ID'].append(gst)
            cval_data['MeanCvalue'].append(None)

    # Create DataFrame with calculated values
    cval_df = pd.DataFrame(cval_data)
    return cval_df



# def back_calculate_Coeff_of_Runoff(flow_file_path, precip_file_path, select_top=10000):
#     """
#     Calculate coefficient of runoff from flow and precipitation data
#     without using datetime functions.

#     Parameters:
#     flow_file_path (str): Path to flow data CSV file
#     precip_file_path (str): Path to precipitation data CSV file
#     select_top (int): Number of top flow values to consider

#     Returns:
#     float: Mean coefficient of runoff value
#     """
#     # Load data
#     f = pd.read_csv(flow_file_path)
#     pr = pd.read_csv(precip_file_path)

#     # Fill missing Hr values with 0 in both datasets
#     f['Hr'] = f['Hr'].fillna(0)
#     f['Min'] = f['Min'].fillna(0)
#     pr['Hr'] = pr['Hr'].fillna(0)
#     pr['Min'] = pr['Min'].fillna(0)

#     # Merge dataframes on common columns (Year, Month, Day, Hr, Min)
#     df = pd.merge(f, pr[['Year', 'Month', 'Day', 'Hr', 'Min', 'PI']], 
#                  on=['Year', 'Month', 'Day', 'Hr', 'Min'],
#                  how='left')
    
#     # Check for missing values in Year, Month, Day columns individually
#     if df['Year'].isna().all() or df['Month'].isna().all() or df['Day'].isna().all():
#         raise ValueError("Upload valid data, Year, Month, Day columns are currently missing.")
    
#     df = df.dropna(subset=['Year'])
    
    
#     # Fill missing PI values with 0
#     df['PI'] = df['PI'].fillna(0)

#     # Create dry period identification
#     df['is_dry'] = df['PI'] == 0

#     # Create groups of consecutive dry/wet periods
#     df['group'] = (df['is_dry'] != df['is_dry'].shift()).cumsum()

#     # Filter for start of wet periods after dry periods of 7 or more hours
#     result_df = df[
#         (df['group'].shift(1) != df['group']) &  # Start of new period
#         (df['is_dry'] == False) &                # Wet period
#         (df.groupby('group')['is_dry'].transform('count').shift(1) >= 7)  # Previous dry period ≥ 7 hours
#     ]

#     # Clean up and prepare final dataset
#     result_df = result_df.drop(columns=['is_dry', 'group'])
#     result_df = result_df[~result_df['Flow'].isna()]
#     result_df = result_df.sort_values(by='Flow', ascending=False)
#     result_df = result_df.head(select_top)

#     # Calculate coefficient of runoff
#     result_df['Cval'] = ((result_df['Flow']/result_df['Area_km2'])* 0.143) / (result_df['PI'] / 2.54)

#     # Filter out invalid values and calculate mean
#     Cvals = result_df['Cval'][~result_df['Cval'].isin([np.nan, np.inf, -np.inf])]

#     return Cvals.mean()


# # ===================================== WS77 =================================================
# flow_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Flow_data/Streamflow_full_series/WS77.csv'
# precip_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Precip_data/Precip_full_series/WS77.csv'
# mean_cval_WS77 = back_calculate_Coeff_of_Runoff(flow_file_path,precip_file_path,select_top=10000)
# print(mean_cval_WS77)

# # ===================================== WS78 =================================================
# flow_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Flow_data/Streamflow_full_series/WS78.csv'
# precip_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Precip_data/Precip_full_series/WS78.csv'
# mean_cval_WS78 = back_calculate_Coeff_of_Runoff(flow_file_path,precip_file_path,select_top=10000)
# print(mean_cval_WS78)
# # ===================================== WS79 =================================================
# flow_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Flow_data/Streamflow_full_series/WS79.csv'
# precip_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Precip_data/Precip_full_series/WS79.csv'
# mean_cval_WS79 = back_calculate_Coeff_of_Runoff(flow_file_path,precip_file_path,select_top=10000)
# print(mean_cval_WS79)
# # ===================================== WS80 =================================================
# flow_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Flow_data/Streamflow_full_series/WS80.csv'
# precip_file_path = '/home/smdsgit/SouravDSGit/CULVERT-Web-App/instance/core_data/WaterData/Precip_data/Precip_full_series/WS80.csv'
# mean_cval_WS80 = back_calculate_Coeff_of_Runoff(flow_file_path,precip_file_path,select_top=10000)
# print(mean_cval_WS80)

# # Define the data
# cval_data = {
#     'Gst_ID': ['WS77', 'WS78', 'WS79', 'WS80'],
#     'MeanCvalue': [mean_cval_WS77, mean_cval_WS78, mean_cval_WS79, mean_cval_WS80]
# }

# # Create the DataFrame
# cval_df = pd.DataFrame(cval_data)

# # Display the DataFrame
# print(cval_df)
# print(cval_df['MeanCvalue'].mean())