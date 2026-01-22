import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, conversion

def init_r_environment(custom_library_path):
    """Initialize R environment with required packages"""
    # Get existing library paths
    current_lib_paths = list(robjects.r(".libPaths()"))

    # Combine existing and new paths, ensuring uniqueness if desired
    # You can choose to prepend or append custom paths
    all_lib_paths = [custom_library_path, '/tmp/R/libraries'] + current_lib_paths

    # Remove duplicates while preserving order (optional, but good practice)
    unique_lib_paths = []
    for path in all_lib_paths:
        if path not in unique_lib_paths:
            unique_lib_paths.append(path)

    # Set the combined library paths
    robjects.r(f".libPaths(c({', '.join(f"'{p}'" for p in unique_lib_paths)}))")

    print("Library paths:", robjects.r(".libPaths()"))

    # Initialize pandas converter
    pandas2ri.activate()
    
    # Import required R packages
    utils = importr("utils")
    required_packages = ['MatrixModels', 'quantreg', 'lme4', 'car', 'rstatix', 
                         'ggpubr','lmomRFA','zoo','extRemes','pracma','nsRFA','dplyr',
                         'evd','ggpubr','ggplot2','lmom','Kendall','openxlsx','sf','nloptr']
    
    for package in required_packages:
        try:
            importr(package)
        except Exception as e:
            print(f"Warning: Could not import {package}: {str(e)}")

def initialize_app(custom_library_path):
    """Initialize the R environment when starting the Flask app"""
    init_r_environment(custom_library_path)