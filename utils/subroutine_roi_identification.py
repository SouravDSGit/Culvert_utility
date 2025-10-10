import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, conversion
from functools import wraps
from rpy2.robjects.conversion import localconverter

# First define the R code
robjects.r(
    r"""
### -----------------------------------------------------------------------------------------------------------------------
## *********************         Function to determine region of influence by following the Homogeneity Test
### -----------------------------------------------------------------------------------------------------------------------
ROI_identification <- function(data_polygon_shapefile_path,
                              ts_dir_path,
                              save_output_roidf_dir_path,
                              GWS_id,
                              cod_ID = 'Point_ID',
                              Test = 'AD'
                              ){


# Read shapefile
geodata <- st_read(data_polygon_shapefile_path)

# Define column names
col_names <- c(cod_ID, 'area_ha', 'GWS_ID', 'AvgELm', 'AvgSL', 'AvgTWI','N30PRin', 
               'ChLen_m', 'OvLen_m', 'TOVmin','TCHmin','TCmin','HySGrpN','CN_val')

# Subset geodata
geodata_subset <- geodata[, col_names]

# Calculate centroid
centroid <- st_centroid(geodata_subset)
centroid_coords <- st_coordinates(centroid)

# Add centroid coordinates to geodata_subset
geodata_subset$Centroid_X <- centroid_coords[, 1]
geodata_subset$Centroid_Y <- centroid_coords[, 2]
print('added centroid coordinates to geodata_subset')

# Convert to dataframe
df <- as.data.frame(geodata_subset)
df <- subset(df, select = c(-geometry))
print('converted to dataframe')

# Extract non-matching GWS_IDs
non_matching_gsts <- na.omit(df$GWS_ID[!is.na(df$GWS_ID) & !(df$GWS_ID %in% GWS_id)])
print(non_matching_gsts)
df$GWS_ID[df$GWS_ID %in% non_matching_gsts] <- NA
print('Replace non-matching values with NA')

# Split gauged and ungauged catchments
p_gauged <- df[!is.na(df$GWS_ID), ]
print('selected all the gauged WSs')
print(p_gauged)

# FIX: Handle duplicate row names by making them unique
cod_gauged_values <- p_gauged[[cod_ID]]
if (any(duplicated(cod_gauged_values))) {
  print('Warning: Duplicate Point_IDs detected in gauged catchments. Making unique...')
  cod_gauged_values <- make.unique(as.character(cod_gauged_values), sep = "_")
}
rownames(p_gauged) <- cod_gauged_values
print('rownames adjusted')
p_gauged <- p_gauged[, !(names(p_gauged) %in% c(cod_ID, "GWS_ID"))]
print(p_gauged)

p_ungauged <- df[is.na(df$GWS_ID), ]

# FIX: Handle duplicate row names for ungauged catchments
cod_ungauged_values <- p_ungauged[[cod_ID]]
if (any(duplicated(cod_ungauged_values))) {
  print('Warning: Duplicate Point_IDs detected in ungauged catchments. Making unique...')
  cod_ungauged_values <- make.unique(as.character(cod_ungauged_values), sep = "_")
}
rownames(p_ungauged) <- cod_ungauged_values
print('rownames adjusted')
p_ungauged <- p_ungauged[, !(names(p_ungauged) %in% c(cod_ID, "GWS_ID"))]
print(p_ungauged)

# Get cod of gauged catchments (using the original values before making unique)
cod_gauged <- df[!is.na(df$GWS_ID), cod_ID]
print('Get cod of gauged catchments')

# Read annual max discharge
gauging_st_name <- df$GWS_ID[!is.na(df$GWS_ID)]
print(gauging_st_name)
print('success: read max discharge data')
x_cod_row <- lapply(seq_along(gauging_st_name), function(i) {
  
  # ------- Define file path for full series
  full_series_file_path <- paste0(ts_dir_path,'/Inst_Streamflow/full_stream_series_',gauging_st_name[i],'.csv')
  print(full_series_file_path)
  # Check if the file exists
  if (file.exists(full_series_file_path)) {
    # Read the CSV into a dataframe
    df_ts <- read.csv(full_series_file_path)
    
    # Compute annual max Flow and divide by Area_km2
    annual_max <- df_ts %>%
      group_by(Year) %>%
      summarise(Flow = max(Flow, na.rm = TRUE),
                Area_km2 = first(Area_km2)) %>%  # Assuming Area_km2 is constant per year
      mutate(Flow = Flow / Area_km2)  # Perform division
  }
  
  
  # ------- Define file path for ams series
  ams_series_file_path <- paste0(ts_dir_path,'/Inst_Streamflow/ams_stream_series_',gauging_st_name[i],'.csv')
  print(ams_series_file_path)
  if (file.exists(ams_series_file_path)) { 
    # Create annual_max with all columns from annual_max_data plus the calculated Flow
    # Read the CSV and ensure it's a data frame
    annual_max_data <- as.data.frame(read.csv(ams_series_file_path))
    annual_max <- annual_max_data
    annual_max$Flow_per_area <- annual_max_data$Flow / annual_max_data$Area_km2
  }
  print('annual_max read')
  print(annual_max)
  cod_i <- rep(cod_gauged[i], length(annual_max$Flow))
  print(cod_i)
  cbind(annual_max$Flow, cod_i)
})
x_cod <- as.data.frame(do.call(rbind, x_cod_row))
print(x_cod)

# ------------------------------------- Region of influence calculation -----------------------
# Apply roi function
roi_df <- apply(p_ungauged, 1, function(row) {
  roi_df_row <- roi(row, p_gauged, cod_gauged)
  roi_df_row[, 1]
})
# Check if as.data.frame(roi_hom) throws an error
error_check <- try(roi_df <- as.data.frame(roi_df), silent = TRUE)

if (inherits(error_check, "try-error")) {  
  # Pad roi_hom vectors to the maximum length
  max_length_df <- max(sapply(roi_df, function(x) if (is.null(x) || all(is.na(x))) 0 else length(x)))
  roi_df_padded <- lapply(roi_df, function(x) {
    if (is.null(x) || all(is.na(x))) {
      return(rep(NA, max_length_df))
    } else {
      current_length_df <- length(x)
      if (current_length_df < max_length_df) {
        c(x, rep(NA, max_length_df - current_length_df))
      } else {
        x
      }
    }
  })
  
  roi_df<- as.data.frame(t(do.call(rbind, roi_df_padded)))
}


colnames(roi_df) <- rownames(p_ungauged)
print("roi_df after padding and conversion:")
print(roi_df)

# ----------------------------------------- ungauged WS homogeneous regions ---------------
# Apply roi.hom function
roi_hom <- apply(p_ungauged, 1, function(row) {
  roi_hom_row <- roi.hom(row, p_gauged, cod_gauged, x_cod[, 1], x_cod[, 2],
                         test = Test, limit = 0.95)
})

# Check if roi_hom is an empty dataframe (0 rows and 0 columns)
if (length(roi_hom) == 0) {
  # Create a new dataframe with the same column names as roi_df, filled with NA
  roi_hom <- data.frame(matrix(NA, nrow = 1, ncol = ncol(roi_df)))
  colnames(roi_hom) <- colnames(roi_df)
  
  # Replace all first-row NAs with the first-row values of roi_df
  roi_hom[1, ] <- roi_df[1, ]
}else{
  
  # Check if as.data.frame(roi_hom) throws an error
  error_check <- try(roi_hom <- as.data.frame(roi_hom), silent = TRUE)
  
  if (inherits(error_check, "try-error")) {  
    # Pad roi_hom vectors to the maximum length
    max_length_hom <- max(sapply(roi_hom, function(x) if (is.null(x) || all(is.na(x))) 0 else length(x)))
    roi_hom_padded <- lapply(roi_hom, function(x) {
      if (is.null(x) || all(is.na(x))) {
        return(rep(NA, max_length_hom))
      } else {
        current_length <- length(x)
        if (current_length < max_length_hom) {
          c(x, rep(NA, max_length_hom - current_length))
        } else {
          x
        }
      }
    })
    
    roi_hom <- as.data.frame(t(do.call(rbind, roi_hom_padded)))
  }
  colnames(roi_hom) <- rownames(p_ungauged)
  
  print("roi_hom after padding and conversion:")
  print(roi_hom)
}

# Identify columns where the first row of roi_hom is NA
na_columns <- which(is.na(roi_hom[1, ]))
# Replace NA values in the first row of roi_hom with the first row of roi_df
if (length(na_columns) > 0) {
  roi_hom[1, na_columns] <- roi_df[1, na_columns]
}
print(roi_hom)



# ----------------------------------------- gauged WS homogeneous regions --------------------------------------
# Apply roi hom for finding the homogeneous region for each gauged WS
roi_hom_gauged <- apply(p_gauged, 1, function(row) {
  roi_hom_gauged_row <- roi.hom(row, p_gauged, cod_gauged, x_cod[, 1], x_cod[, 2],
                                test = Test, limit = 0.95)
})

# Check if roi_hom_gauged is empty (0 rows and 0 columns)
if (length(roi_hom_gauged) == 0) {
  # Create a DataFrame with row names as column names and NA values
  roi_hom_gauged <- data.frame(matrix(NA, nrow = 1, ncol = length(rownames(p_gauged))))
  colnames(roi_hom_gauged) <- rownames(p_gauged)
}else{
  
  # Check if as.data.frame(roi_hom) throws an error
  error_check_gauged <- try(roi_hom_gauged <- as.data.frame(roi_hom_gauged), silent = TRUE)
  
  if (inherits(error_check_gauged, "try-error")) {
    # Pad roi_hom vectors to the maximum length
    max_length_hom_gauged <- max(sapply(roi_hom_gauged, function(x) if (is.null(x) || all(is.na(x))) 0 else length(x)))
    roi_hom_padded_gauged <- lapply(roi_hom_gauged, function(x) {
      if (is.null(x) || all(is.na(x))) {
        return(rep(NA, max_length_hom_gauged))
      } else {
        current_length_gauged <- length(x)
        if (current_length_gauged < max_length_hom_gauged) {
          c(x, rep(NA, max_length_hom_gauged - current_length_gauged))
        } else {
          x
        }
      }
    })
    
    roi_hom_gauged <- as.data.frame(t(do.call(rbind, roi_hom_padded_gauged)))
  }
  colnames(roi_hom_gauged) <- rownames(p_gauged)
  
  print("roi_hom_gauged after padding and conversion:")
  print(roi_hom_gauged)
}

# Identify columns where the first row of roi_hom_gauged is NA
na_columns <- which(is.na(roi_hom_gauged[1, ]))

# Replace NA values in the first row with the corresponding column names
if (length(na_columns) > 0) {
  roi_hom_gauged[1, na_columns] <- colnames(roi_hom_gauged)[na_columns]
}

  #### -----------------------------------------------------------------------------------------------
  # Save roi_hom dataframe
  write.csv(roi_hom, paste0(save_output_roidf_dir_path,"/roi_hom.csv"), row.names = FALSE)
  
  # Save roi_hom_gauged dataframe
  write.csv(roi_hom_gauged, paste0(save_output_roidf_dir_path,"/roi_hom_gauged.csv"), row.names = FALSE)
  
  # Save roi_df dataframe
  write.csv(roi_df, paste0(save_output_roidf_dir_path,"/roi_index.csv"), row.names = FALSE)
  
  ### ------------------------------------------------------------------------------------------------

  # Don't return anything
  invisible(NULL)
  
}
"""
)

def with_r_context(func):
    """Decorator to ensure proper R conversion context"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            return func(*args, **kwargs)
    return wrapper

@with_r_context
def roi_identification(
    data_polygon_shapefile_path,
    ts_dir_path,
    save_output_roidf_dir_path,
    GWS_id,
    cod_ID="Point_ID",
    Test="AD",
):
    try:
        GWS_id_r = "c(" + ", ".join(f"'{gid}'" for gid in GWS_id) + ")"
        
        # Execute the R function without capturing the result
        robjects.r(
            f"""
            tryCatch({{
                ROI_identification(
                    data_polygon_shapefile_path = '{data_polygon_shapefile_path}',
                    ts_dir_path = '{ts_dir_path}',
                    save_output_roidf_dir_path = '{save_output_roidf_dir_path}',
                    GWS_id = {GWS_id_r},
                    cod_ID = '{cod_ID}',
                    Test = '{Test}'
                )
            }}, error = function(e) {{
                print(paste("Error in R execution:", e$message))
                stop(e$message)
            }})
            """
        )
        
    except Exception as e:
        raise Exception(f"Error in ROI identification: {str(e)}")