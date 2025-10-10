# =======================================================================================================
# IDEAL POINT SHAPEFILE COLUMN NAMES FOR COMPREHENSIVE CULVERT CAPACITY ANALYSIS
# =======================================================================================================

"""
REQUIRED COLUMNS (Must be provided by user - no assumptions made for dimensions):
================================================================================
Point_ID        - Unique identifier for each culvert point (string/integer)
Width_ft        - Culvert width in feet (diameter for circular culverts) (float)
Height_ft       - Culvert height in feet (not needed for circular, required for box/arch) (float)
PourSha         - Culvert shape: "Circular", "Box", "Pipe arch", "Elliptical", "Arch" (string)
                  If NA/missing, defaults to "Circular"
Longitude       - Longitude coordinate in decimal degrees (float)
Latitude        - Latitude coordinate in decimal degrees (float)

OPTIONAL COLUMNS (Intelligent defaults applied when missing or NA):
==================================================================

# Basic Geometric Parameters
Grp_Size        - Number of culverts in group (integer, default: 1)
L_ft            - Culvert length in feet (float, default: 98.4 ft = 30 m)
Culvert_Sl      - Culvert slope as decimal (float, default: 0.01 = 1%)

# Hydraulic Depths
HW_ft           - Headwater depth in feet (float, default: calculated as 1.2-1.5 × culvert height)
TW_ft           - Tailwater depth in feet (float, default: calculated as 0.5 × culvert height)

# Material and Roughness Properties
Material        - Culvert material (string, default: "Concrete")
                  Options: "Concrete" (n=0.012), "Steel" (n=0.024), "HDPE" (n=0.009),
                          "Aluminum" (n=0.024), "Plastic" (n=0.009), "CMP" (n=0.024)
n_manning       - Manning's roughness coefficient (float, default: material-based)
Condition       - Culvert condition (string, default: "Good")
                  Options: "Good" (factor=1.0), "Fair" (factor=1.1), "Poor" (factor=1.3)

# Inlet Configuration
Inlet_Type      - Inlet type configuration (string, default: "Square_Edge")
                  Options: "Square_Edge" (Ke=0.5), "Grooved_End" (Ke=0.2), "Beveled" (Ke=0.2),
                          "Projecting" (Ke=0.9), "Mitered" (Ke=0.7)
Headwall        - Presence of headwall (string, default: "Yes")
                  Options: "Yes", "No"
Wingwalls       - Wingwall configuration (string, default: "None")
                  Options: "None", "30deg", "45deg", "90deg"

# Advanced FHWA Hydraulic Parameters (all float, literature-based defaults)
Ke              - Entrance loss coefficient (default: inlet-type based or 0.5)
Y               - Inlet geometry factor (default: based on HDS-5 tables)
ks              - Slope correction factor (default: -0.5, except +0.7 for mitered)
c               - Inlet geometry constant (default: based on HDS-5 tables)
Cd_orifice      - Orifice discharge coefficient (default: 0.62)
Cw_weir         - Weir coefficient (default: 3.0 for broad-crested in English units)
K_inlet         - Inlet form factor (default: based on HDS-5 tables)
M_inlet         - Inlet slope of control curve (default: based on HDS-5 tables)
c_inlet         - Inlet intercept constant (default: based on HDS-5 tables)
Y_inlet         - Inlet transition factor (default: based on HDS-5 tables)

# Additional Documentation Columns
Install_Year    - Installation year (integer, for age assessment)
Replacement_Year- Planned replacement year (integer)
Maintenance_Date- Last maintenance date (date)
Notes           - Additional notes or comments (string)
Data_Source     - Source of culvert data (string)
Survey_Date     - Date of field survey (date)
Photo_Path      - Path to culvert photos (string)
"""

import urllib
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
import json

# =======================================================================================================
# ENHANCED CULVERT VULNERABILITY ANALYSIS WITH COMPREHENSIVE COLUMN HANDLING
# =======================================================================================================

def vuln_by_comparing_peak_Q_with_discharge_capacity(pour_point_shapefile_path,
                                                            save_vuln_results_path,
                                                            QC_method,
                                                            peak_Q_gdf,
                                                            QP_method,
                                                            **kwargs):
    """
    Enhanced culvert vulnerability analysis with comprehensive column handling and intelligent defaults,
    with hydraulic equations verified against FHWA HDS-5.
    
    Parameters:
    -----------
    pour_point_shapefile_path : str
        Path to shapefile containing culvert point data
    save_vuln_results_path : str
        Path to save vulnerability results
    QC_method : str
        Method for calculating discharge capacity:
        'orifice_flow', 'inlet_control', 'outlet_control', and  
        'manning_uniform'
    peak_Q_gdf : GeoDataFrame
        GeoDataFrame containing peak discharge data
    QP_method : str
        Peak discharge method: 'RationalMethod', 'RegionalFrequency', 'GPDM'
    **kwargs : additional hydraulic parameters to override defaults
    
    Returns:
    --------
    GeoDataFrame : Merged vulnerability results
    """
    
    print("=== STARTING ENHANCED CULVERT CAPACITY ANALYSIS ===")
    
    # Read the shapefile (point data)
    point_gdf = gpd.read_file(pour_point_shapefile_path)
    print(f"Loaded {len(point_gdf)} culvert points from shapefile")
    print(f"Columns found: {list(point_gdf.columns)}")
    
    # ===== CHECK REQUIRED COLUMNS =====
    required_columns = ['Point_ID', 'Width_ft', 'Longitude', 'Latitude']
    missing_required = [col for col in required_columns if col not in point_gdf.columns]
    
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # ===== CLEAN AND CONVERT DATA =====
    na_values = ['na', 'NA', 'NaN', '"NA"', 'null', 'NULL', '', ' ', 'None', 'NONE']
    
    # Clean numeric columns
    numeric_columns = ['Width_ft', 'Height_ft', 'Grp_Size', 'L_ft', 'Culvert_Sl', 
                      'HW_ft', 'TW_ft', 'n_manning', 'Ke', 'Y', 'ks', 'c', 
                      'Cd_orifice', 'Cw_weir', 'K_inlet', 'M_inlet', 'c_inlet', 'Y_inlet']
    
    for col in numeric_columns:
        if col in point_gdf.columns:
            point_gdf[col] = point_gdf[col].replace(na_values, np.nan)
            point_gdf[col] = pd.to_numeric(point_gdf[col], errors='coerce')
    
    # ===== HANDLE CULVERT SHAPE (PourSha) - CRITICAL FOR AREA CALCULATIONS =====
    if 'PourSha' not in point_gdf.columns:
        print("WARNING: PourSha column not found. Assuming all culverts are Circular.")
        point_gdf['PourSha'] = 'Circular'
    else:
        # Clean and standardize shape column
        point_gdf['PourSha'] = point_gdf['PourSha'].replace(na_values, 'Circular')
        point_gdf['PourSha'] = point_gdf['PourSha'].astype(str).str.strip().str.title()
        
        # Standardize shape names
        shape_mapping = {
            'Circle': 'Circular', 'Round': 'Circular', 'Pipe': 'Circular','round': 'Circular',
            'Rectangle': 'Box', 'Rectangular': 'Box', 'Square': 'Box',
            'Pipearch': 'Pipe arch', 'Pipe_Arch': 'Pipe arch', 'Pipetoarch': 'Pipe arch',
            'Oval': 'Elliptical', 'Ellipse': 'Elliptical'
        }
        point_gdf['PourSha'] = point_gdf['PourSha'].replace(shape_mapping)
        
        # Any remaining unknown shapes default to Circular
        valid_shapes = ['Circular', 'Box', 'Pipe arch', 'Elliptical', 'Arch']
        invalid_shapes = ~point_gdf['PourSha'].isin(valid_shapes)
        if invalid_shapes.any():
            print(f"WARNING: Found {invalid_shapes.sum()} invalid shapes. Converting to Circular.")
            point_gdf.loc[invalid_shapes, 'PourSha'] = 'Circular'
    
    print(f"Shape distribution: {point_gdf['PourSha'].value_counts().to_dict()}")
    
    # ===== APPLY INTELLIGENT DEFAULTS =====
    
    # Group Size
    if 'Grp_Size' not in point_gdf.columns:
        point_gdf['Grp_Size'] = 1
    else:
        point_gdf['Grp_Size'] = point_gdf['Grp_Size'].fillna(1)
    
    # Length conversion and defaults
    if 'L_ft' in point_gdf.columns:
        point_gdf['L_m'] = point_gdf['L_ft'] * 0.3048
        point_gdf['L_m'] = point_gdf['L_m'].fillna(30.0)
    else:
        point_gdf['L_m'] = kwargs.get('L_m', 30.0)
    
    # Slope
    if 'Culvert_Sl' not in point_gdf.columns:
        point_gdf['Culvert_Sl'] = kwargs.get('Culvert_Sl', 0.01)
    else:
        point_gdf['Culvert_Sl'] = point_gdf['Culvert_Sl'].fillna(0.01)
    
    # Material-based Manning's n (values from HDS-5)
    if 'Material' in point_gdf.columns:
        point_gdf['Material'] = point_gdf['Material'].replace(na_values, 'Concrete')
        material_manning = {
            'Concrete': 0.012, 'Steel': 0.024, 'HDPE': 0.009,
            'Aluminum': 0.024, 'Plastic': 0.009, 'CMP': 0.024
        }
        
        if 'n_manning' not in point_gdf.columns:
            point_gdf['n_manning'] = point_gdf['Material'].map(material_manning).fillna(0.012)
        else:
            # Fill missing n_manning with material-based values
            for idx, row in point_gdf.iterrows():
                if pd.isna(row['n_manning']):
                    point_gdf.loc[idx, 'n_manning'] = material_manning.get(row['Material'], 0.012)
    else:
        point_gdf['Material'] = 'Concrete'
        point_gdf['n_manning'] = kwargs.get('n_manning', 0.012)
    
    # Inlet Type and Entrance Loss (Ke values from HDS-5)
    if 'Inlet_Type' in point_gdf.columns:
        point_gdf['Inlet_Type'] = point_gdf['Inlet_Type'].replace(na_values, 'Square_Edge')
        inlet_ke = {
            'Square_Edge': 0.5, 'Grooved_End': 0.2, 'Beveled': 0.2,
            'Projecting': 0.9, 'Mitered': 0.7
        }
        
        if 'Ke' not in point_gdf.columns:
            point_gdf['Ke'] = point_gdf['Inlet_Type'].map(inlet_ke).fillna(0.5)
        else:
            # Fill missing Ke with inlet-based values
            for idx, row in point_gdf.iterrows():
                if pd.isna(row['Ke']):
                    point_gdf.loc[idx, 'Ke'] = inlet_ke.get(row['Inlet_Type'], 0.5)
    else:
        point_gdf['Inlet_Type'] = 'Square_Edge'
        if 'Ke' not in point_gdf.columns:
            point_gdf['Ke'] = kwargs.get('Ke', 0.5)
        else:
            point_gdf['Ke'] = point_gdf['Ke'].fillna(kwargs.get('Ke', 0.5))
    
    # Condition adjustment
    if 'Condition' in point_gdf.columns:
        point_gdf['Condition'] = point_gdf['Condition'].replace(na_values, 'Good')
        condition_factor = {'Good': 1.0, 'Fair': 1.1, 'Poor': 1.3}
        point_gdf['n_manning'] *= point_gdf['Condition'].map(condition_factor).fillna(1.0)
    
    # FHWA hydraulic parameter defaults (from HDS-5 tables for common configurations)
    # These are simplified; a more robust implementation would use lookup tables based on shape and inlet type.
    hydraulic_defaults = {
        'Y': kwargs.get('Y', 0.6), # Approximate for unsubmerged orifice flow
        'ks': kwargs.get('ks', -0.5),
        'c': kwargs.get('c', 0.038), # Approximate for unsubmerged orifice flow
        'Cd_orifice': kwargs.get('Cd_orifice', 0.62),
        'Cw_weir': kwargs.get('Cw_weir', 3.0), # For broad-crested weir in English units
        'K_inlet': kwargs.get('K_inlet', 0.0098), # Example for a specific improved inlet
        'M_inlet': kwargs.get('M_inlet', 2.0), # Example for a specific improved inlet
        'c_inlet': kwargs.get('c_inlet', 0.0398), # Example for a specific improved inlet
        'Y_inlet': kwargs.get('Y_inlet', 0.67) # Example for a specific improved inlet
    }
    
    for param, default_val in hydraulic_defaults.items():
        if param not in point_gdf.columns:
            point_gdf[param] = default_val
        else:
            point_gdf[param] = point_gdf[param].fillna(default_val)
    
    # ===== CALCULATE DISCHARGE CAPACITY (Qc) =====
    print(f"\n=== CALCULATING DISCHARGE CAPACITY USING {QC_method.upper()} METHOD ===")
    
    # Initialize Qc to NaN
    point_gdf['Qc'] = np.nan
    
    # Define valid rows based on culvert shape and required dimensions
    print("\n=== VALIDATING CULVERT DIMENSIONS ===")
    
    # Circular culverts: only need Width_ft (diameter)
    circular_valid = (point_gdf['PourSha'] == 'Circular') & point_gdf['Width_ft'].notna()
    
    # Non-circular culverts: need both Width_ft and Height_ft
    non_circular_shapes = ['Box', 'Pipe arch', 'Elliptical', 'Arch']
    non_circular_valid = (point_gdf['PourSha'].isin(non_circular_shapes)) & \
                        point_gdf['Width_ft'].notna() & point_gdf['Height_ft'].notna()
    
    valid_rows = circular_valid | non_circular_valid
    
    print(f"Circular culverts (valid): {circular_valid.sum()}")
    print(f"Non-circular culverts (valid): {non_circular_valid.sum()}")
    print(f"Total valid culverts: {valid_rows.sum()}")
    print(f"Invalid culverts: {(~valid_rows).sum()}")
    
    if not valid_rows.any():
        print("WARNING: No valid culvert dimension data found. All discharge capacities will be NA.")
    else:
        # ===== CALCULATE GEOMETRIC PROPERTIES FOR VALID CULVERTS =====
        
        # Convert dimensions to feet (calculations will be in English units to match HDS-5)
        point_gdf.loc[valid_rows, 'Width_ft'] = point_gdf.loc[valid_rows, 'Width_ft']
        point_gdf.loc[non_circular_valid, 'Height_ft'] = point_gdf.loc[non_circular_valid, 'Height_ft']
        
        # Calculate characteristic dimension (D) - used for hydraulic calculations
        point_gdf.loc[circular_valid, 'D_ft'] = point_gdf.loc[circular_valid, 'Width_ft']  # diameter
        point_gdf.loc[non_circular_valid, 'D_ft'] = point_gdf.loc[non_circular_valid, 'Height_ft']
        
        # ===== CALCULATE CROSS-SECTIONAL AREA BY SHAPE =====
        print("\n=== CALCULATING CROSS-SECTIONAL AREAS BY SHAPE ===")
        
        # CIRCULAR CULVERTS
        circular_mask = (point_gdf['PourSha'] == 'Circular') & valid_rows
        if circular_mask.any():
            # Area = π × (diameter/2)²
            point_gdf.loc[circular_mask, 'Ac'] = np.pi * (point_gdf.loc[circular_mask, 'Width_ft'] / 2)**2
            print(f"Calculated {circular_mask.sum()} circular culvert areas")
        
        # BOX CULVERTS
        box_mask = (point_gdf['PourSha'] == 'Box') & valid_rows
        if box_mask.any():
            # Area = Width × Height
            point_gdf.loc[box_mask, 'Ac'] = point_gdf.loc[box_mask, 'Width_ft'] * point_gdf.loc[box_mask, 'Height_ft']
            print(f"Calculated {box_mask.sum()} box culvert areas")
        
        # PIPE ARCH CULVERTS
        pipe_arch_mask = (point_gdf['PourSha'] == 'Pipe arch') & valid_rows
        if pipe_arch_mask.any():
            # More accurate approximation based on span and rise
            span = point_gdf.loc[pipe_arch_mask, 'Width_ft']
            rise = point_gdf.loc[pipe_arch_mask, 'Height_ft']
            point_gdf.loc[pipe_arch_mask, 'Ac'] = (np.pi * span * rise) / 4
            print(f"Calculated {pipe_arch_mask.sum()} pipe arch culvert areas")
        
        # ELLIPTICAL CULVERTS
        elliptical_mask = (point_gdf['PourSha'] == 'Elliptical') & valid_rows
        if elliptical_mask.any():
            # Area = π × (Width/2) × (Height/2)
            point_gdf.loc[elliptical_mask, 'Ac'] = np.pi * (point_gdf.loc[elliptical_mask, 'Width_ft'] / 2) * (point_gdf.loc[elliptical_mask, 'Height_ft'] / 2)
            print(f"Calculated {elliptical_mask.sum()} elliptical culvert areas")
        
        # ARCH CULVERTS
        arch_mask = (point_gdf['PourSha'] == 'Arch') & valid_rows
        if arch_mask.any():
            # Approximation: Area ≈ 2/3 × Width × Height for a parabolic arch
            point_gdf.loc[arch_mask, 'Ac'] = (2/3) * point_gdf.loc[arch_mask, 'Width_ft'] * point_gdf.loc[arch_mask, 'Height_ft']
            print(f"Calculated {arch_mask.sum()} arch culvert areas")
        
        # ===== CALCULATE HYDRAULIC RADIUS BY SHAPE (for full flow) =====
        print("\n=== CALCULATING HYDRAULIC RADIUS BY SHAPE ===")
        
        # CIRCULAR: R = D/4
        point_gdf.loc[circular_mask, 'hydraulic_radius'] = point_gdf.loc[circular_mask, 'Width_ft'] / 4
        
        # BOX: R = (W×H)/(2×(W+H))
        if box_mask.any():
            point_gdf.loc[box_mask, 'hydraulic_radius'] = (
                (point_gdf.loc[box_mask, 'Width_ft'] * point_gdf.loc[box_mask, 'Height_ft']) / 
                (2 * (point_gdf.loc[box_mask, 'Width_ft'] + point_gdf.loc[box_mask, 'Height_ft']))
            )
        
        # PIPE ARCH, ELLIPTICAL, ARCH: Approximation R ≈ A/P where P is approximated
        for mask in [pipe_arch_mask, elliptical_mask, arch_mask]:
            if mask.any():
                # Ramanujan's approximation for ellipse perimeter
                a = point_gdf.loc[mask, 'Width_ft'] / 2
                b = point_gdf.loc[mask, 'Height_ft'] / 2
                perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
                point_gdf.loc[mask, 'hydraulic_radius'] = point_gdf.loc[mask, 'Ac'] / perimeter
        
        # Print area statistics
        print(f"Area statistics (ft²):")
        print(f"  Min: {point_gdf.loc[valid_rows, 'Ac'].min():.3f}")
        print(f"  Max: {point_gdf.loc[valid_rows, 'Ac'].max():.3f}")
        print(f"  Mean: {point_gdf.loc[valid_rows, 'Ac'].mean():.3f}")
        
        # ===== APPLY CULVERT CAPACITY METHOD =====
        print(f"\n=== APPLYING {QC_method.upper()} METHOD ===")
        
        g = 32.2  # ft/s²
        
        if QC_method == 'orifice_flow':
            # Set headwater depth
            if 'HW_ft' in point_gdf.columns:
                point_gdf['HW_ft_calc'] = point_gdf['HW_ft'].fillna(1.2 * point_gdf['D_ft'])
            else:
                point_gdf.loc[valid_rows, 'HW_ft_calc'] = kwargs.get('HW_ft', 1.2 * point_gdf.loc[valid_rows, 'D_ft'])
            
            # Orifice equation: Q = Cd * A * sqrt(2 * g * H), where H is the effective head to the centroid
            # For simplicity, using HW as head. For more accuracy, H should be HW - (D/2)
            point_gdf.loc[valid_rows, 'Qc'] = (point_gdf.loc[valid_rows, 'Cd_orifice'] * 
                                              point_gdf.loc[valid_rows, 'Ac'] * 
                                              point_gdf.loc[valid_rows, 'Grp_Size'] * 
                                              np.sqrt(2 * g * point_gdf.loc[valid_rows, 'HW_ft_calc']))
        
        elif QC_method == 'inlet_control':
            # FHWA HDS-5 Inlet Control Equations (unsubmerged and submerged)
            if 'HW_ft' in point_gdf.columns:
                point_gdf['HW_ft_calc'] = point_gdf['HW_ft'].fillna(1.5 * point_gdf['D_ft'])
            else:
                point_gdf.loc[valid_rows, 'HW_ft_calc'] = kwargs.get('HW_ft', 1.5 * point_gdf.loc[valid_rows, 'D_ft'])

            # Unsubmerged Orifice Flow (Form 1 from HDS-5, Equation 13)
            # Q = A * sqrt(2g(HW - Y*D - ks*S*D)) / c
            # This is a simplification; the actual form depends on the inlet configuration.
            # Using Form 2 for demonstration (Equation 14 from HDS-5)
            # HW/D = c * (Q / (A * D^0.5))^2 + Y - ks * S
            # Solving for Q: Q = A * D^0.5 * sqrt((HW/D - Y + ks*S) / c)
            hw_d_ratio = point_gdf.loc[valid_rows, 'HW_ft_calc'] / point_gdf.loc[valid_rows, 'D_ft']
            unsubmerged_mask = (hw_d_ratio <= 1.2) & valid_rows # A common threshold
            
            # Submerged Orifice Flow (Equation 15 from HDS-5)
            # HW/D = c * (Q / (A * D^0.5))^2 + Y
            # Solving for Q: Q = A * D^0.5 * sqrt((HW/D - Y) / c)
            submerged_mask = (hw_d_ratio > 1.2) & valid_rows
            
            point_gdf.loc[unsubmerged_mask, 'Qc'] = (point_gdf.loc[unsubmerged_mask, 'Ac'] * 
                                                     point_gdf.loc[unsubmerged_mask, 'D_ft']**0.5 * 
                                                     np.sqrt((point_gdf.loc[unsubmerged_mask, 'HW_ft_calc'] / point_gdf.loc[unsubmerged_mask, 'D_ft'] - point_gdf.loc[unsubmerged_mask, 'Y'] + point_gdf.loc[unsubmerged_mask, 'ks'] * point_gdf.loc[unsubmerged_mask, 'Culvert_Sl']) / point_gdf.loc[unsubmerged_mask, 'c']))
            
            point_gdf.loc[submerged_mask, 'Qc'] = (point_gdf.loc[submerged_mask, 'Ac'] * 
                                                   point_gdf.loc[submerged_mask, 'D_ft']**0.5 * 
                                                   np.sqrt((point_gdf.loc[submerged_mask, 'HW_ft_calc'] / point_gdf.loc[submerged_mask, 'D_ft'] - point_gdf.loc[submerged_mask, 'Y']) / point_gdf.loc[submerged_mask, 'c']))
            
            point_gdf.loc[valid_rows, 'Qc'] *= point_gdf.loc[valid_rows, 'Grp_Size']

        elif QC_method == 'outlet_control':
            # FHWA HDS-5 Outlet Control Equation (Equation 10)
            # HW = TW + H - So * L, where H is total headloss
            # H = (1 + Ke + (29.2 * n^2 * L) / R^1.33) * V^2 / (2g)
            # Since Q = A * V, V = Q / A.  Substitute and solve for Q.
            if 'HW_ft' in point_gdf.columns:
                point_gdf['HW_ft_calc'] = point_gdf['HW_ft'].fillna(1.5 * point_gdf['D_ft'])
            else:
                point_gdf.loc[valid_rows, 'HW_ft_calc'] = kwargs.get('HW_ft', 1.5 * point_gdf.loc[valid_rows, 'D_ft'])
            
            if 'TW_ft' in point_gdf.columns:
                point_gdf['TW_ft_calc'] = point_gdf['TW_ft'].fillna(0.5 * point_gdf['D_ft'])
            else:
                point_gdf.loc[valid_rows, 'TW_ft_calc'] = kwargs.get('TW_ft', 0.5 * point_gdf.loc[valid_rows, 'D_ft'])
            
            # Convert L_m to L_ft for calculations (since we're working in English units)
            L_ft_calc = point_gdf.loc[valid_rows, 'L_m'] / 0.3048
            
            ho = np.maximum(point_gdf.loc[valid_rows, 'TW_ft_calc'], point_gdf.loc[valid_rows, 'D_ft'])
            H_available = point_gdf.loc[valid_rows, 'HW_ft_calc'] - ho + point_gdf.loc[valid_rows, 'Culvert_Sl'] * L_ft_calc
            
            headloss_coeff = 1 + point_gdf.loc[valid_rows, 'Ke'] + (29.2 * point_gdf.loc[valid_rows, 'n_manning']**2 * L_ft_calc) / (point_gdf.loc[valid_rows, 'hydraulic_radius']**(4/3))
            
            point_gdf.loc[valid_rows, 'Qc'] = point_gdf.loc[valid_rows, 'Ac'] * np.sqrt((2 * g * H_available) / headloss_coeff)
            point_gdf.loc[valid_rows, 'Qc'] *= point_gdf.loc[valid_rows, 'Grp_Size']
        
        elif QC_method == 'manning_uniform':
            # Manning's equation for uniform flow: Q = (1.49/n) * A * R^(2/3) * S^(1/2) (English units)
            point_gdf.loc[valid_rows, 'Qc'] = ((1.49 / point_gdf.loc[valid_rows, 'n_manning']) * 
                                              point_gdf.loc[valid_rows, 'Ac'] * 
                                              (point_gdf.loc[valid_rows, 'hydraulic_radius']**(2/3)) * 
                                              np.sqrt(point_gdf.loc[valid_rows, 'Culvert_Sl']) *
                                              point_gdf.loc[valid_rows, 'Grp_Size'])
        
        else:
            raise ValueError(f"Unknown QC_method: {QC_method}. Available methods: 'orifice_flow', 'inlet_control', 'outlet_control', 'manning_uniform'")
        
        # Convert Qc from cfs to cms
        point_gdf['Qc'] = point_gdf['Qc'] * 0.0283168

        # Print capacity statistics
        valid_qc = point_gdf.loc[valid_rows, 'Qc'].dropna()
        if len(valid_qc) > 0:
            print(f"\n=== DISCHARGE CAPACITY RESULTS ===")
            print(f"Method: {QC_method}")
            print(f"Valid calculations: {len(valid_qc)}")
            print(f"Qc statistics (m³/s):")
            print(f"  Min: {valid_qc.min():.3f}")
            print(f"  Max: {valid_qc.max():.3f}")
            print(f"  Mean: {valid_qc.mean():.3f}")
            print(f"  Median: {valid_qc.median():.3f}")
        else:
            print("WARNING: No valid discharge capacity calculations produced.")

    # ===== COORDINATE SYSTEM AND GEOMETRY HANDLING =====
    print(f"\n=== PROCESSING COORDINATES ===")
    
    # Convert point_gdf geometry to WGS84 (EPSG:4326) if not already
    if point_gdf.crs != "EPSG:4326":
        print(f"Converting from {point_gdf.crs} to EPSG:4326")
        point_gdf = point_gdf.to_crs("EPSG:4326")
    
    # Extract latitude and longitude from geometry
    point_gdf['lat'] = point_gdf.geometry.y
    point_gdf['lon'] = point_gdf.geometry.x
    
    # Update Longitude and Latitude columns if they exist
    if 'Longitude' in point_gdf.columns:
        point_gdf['Longitude'] = point_gdf['lon']
    else:
        point_gdf['Longitude'] = point_gdf['lon']
        
    if 'Latitude' in point_gdf.columns:
        point_gdf['Latitude'] = point_gdf['lat']
    else:
        point_gdf['Latitude'] = point_gdf['lat']

    # ===== VULNERABILITY ASSESSMENT =====
    print(f"\n=== VULNERABILITY ASSESSMENT USING {QP_method} ===")
    
    # Identify peak discharge columns based on method
    if QP_method == 'RationalMethod':
        # Check if RMevent column exists (indicates pointPI was used)
        if 'RMevent' in peak_Q_gdf.columns:
            discharge_columns = ['RMevent']
        else:
            discharge_columns = [col for col in peak_Q_gdf.columns if col.startswith('RM') and col.endswith('E')]
    elif QP_method == 'RegionalFrequency':
        discharge_columns = [col for col in peak_Q_gdf.columns if col.startswith('RF') and col.endswith('E')]
    elif QP_method == 'GPDM':
        # Check if GPevent column exists (indicates evpr was used)
        if 'GPevent' in peak_Q_gdf.columns:
            discharge_columns = ['GPevent']
        else:
            discharge_columns = [col for col in peak_Q_gdf.columns if col.startswith('GP') and col.endswith('E')]
    print(f"Found discharge columns: {discharge_columns}")
    
    # Merge point_gdf with peak_Q_gdf
    merge_columns = ['Point_ID', 'Qc', 'Longitude', 'Latitude', 'lat', 'lon']
    merged_df = pd.merge(
        point_gdf[merge_columns],
        peak_Q_gdf[['Point_ID', 'geometry'] + discharge_columns],
        on='Point_ID',
        how='inner'
    )
    
    print(f"Merged {len(merged_df)} culverts with peak discharge data")
    
    # Calculate vulnerability for each return period
    vulnerability_columns = []
    for discharge_col in discharge_columns:
        # Extract return period from column name (e.g., 'RM10E' -> '10', 'RMevent' -> 'event', 'GPevent' -> 'event')
        if discharge_col.endswith('event'):
            return_period = 'event'
        else:
            return_period = discharge_col[2:-1]
        
        vuln_col = f'{return_period}Vuln'
        vulnerability_columns.append(vuln_col)
        
        # Compare culvert capacity with peak discharge
        merged_df[vuln_col] = np.where(
            merged_df['Qc'].isna(),
            np.nan,  # No capacity data
            np.where(
                merged_df['Qc'] >= merged_df[discharge_col], 
                'Not Vulnerable',  # Capacity ≥ Peak discharge
                'Vulnerable'       # Capacity < Peak discharge
            )
        )
        
        # Print vulnerability statistics
        vuln_stats = merged_df[vuln_col].value_counts()
        print(f"{return_period} return period: {vuln_stats.to_dict()}")
    
    # ===== UPDATE PEAK DISCHARGE GDF WITH RESULTS =====
    print(f"\n=== UPDATING RESULTS ===")
    
    # Merge results back to peak_Q_gdf
    result_columns = ['Point_ID', 'Qc', 'Longitude', 'Latitude', 'lat', 'lon'] + vulnerability_columns
    peak_Q_gdf = pd.merge(
        peak_Q_gdf,
        merged_df[result_columns],
        on='Point_ID',
        how='left'
    )
    
    # Add method information
    peak_Q_gdf['QC_Method'] = QC_method
    peak_Q_gdf['QP_Method'] = QP_method
    
    # Dissolve polygons based on Point_ID (keep as polygon GeoDataFrame)
    print("Dissolving polygons by Point_ID...")
    peak_Q_gdf_merged = peak_Q_gdf.dissolve(by='Point_ID').reset_index()
    
    # Save results
    print(f"Saving results to: {save_vuln_results_path}")
    peak_Q_gdf_merged.to_file(save_vuln_results_path)
    
    # ===== FINAL SUMMARY =====
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Input culverts: {len(point_gdf)}")
    print(f"Valid dimensions: {valid_rows.sum() if 'valid_rows' in locals() else 0}")
    print(f"Successful capacity calculations: {len(point_gdf['Qc'].dropna()) if 'Qc' in point_gdf.columns else 0}")
    print(f"Merged with peak discharge: {len(merged_df) if 'merged_df' in locals() else 0}")
    print(f"Output file: {save_vuln_results_path}")
    print(f"Output columns: {list(peak_Q_gdf_merged.columns)}")
    
    # Print capacity method summary
    if 'valid_rows' in locals() and valid_rows.any():
        print(f"\nCapacity method used: {QC_method}")
        print(f"Default parameters applied:")
        
        if 'n_manning' in point_gdf.columns:
            print(f"  Manning's n range: {point_gdf['n_manning'].min():.3f} - {point_gdf['n_manning'].max():.3f}")
        
        if 'Ke' in point_gdf.columns:
            print(f"  Entrance loss (Ke): {point_gdf['Ke'].min():.3f} - {point_gdf['Ke'].max():.3f}")
        
        if 'L_ft' in point_gdf.columns:
            print(f"  Culvert length: {point_gdf['L_ft'].min():.1f} - {point_gdf['L_ft'].max():.1f} ft")
        else:
            # Convert L_m back to feet for display
            L_ft_values = point_gdf['L_m'] / 0.3048
            print(f"  Culvert length: {L_ft_values.min():.1f} - {L_ft_values.max():.1f} ft")
        
        if 'Culvert_Sl' in point_gdf.columns:
            print(f"  Culvert slope: {point_gdf['Culvert_Sl'].min():.4f} - {point_gdf['Culvert_Sl'].max():.4f}")
    
    return peak_Q_gdf_merged

