from functools import wraps
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import pandas as pd
import geopandas as gpd
import os




# Run R code
robjects.r(
  
  r"""


######################################################################################################################
# Function: detect_and_remove_outliers
# Purpose: Detects and removes outliers from a specified column in a dataframe using either z-score or IQR method.
# Inputs:
#   - data: The dataframe containing the data.
#   - column_name: The name of the column to check for outliers.
#   - method: Outlier detection method ('z-score' or 'IQR').
#   - z_threshold: Threshold for z-score method (default: 3).
# Output:
#   - The cleaned dataframe with outliers removed.
######################################################################################################################
## Function to remove outliers
detect_and_remove_outliers <- function(data, column_name, method = c("Zscore","None","IQR"), z_threshold = 3) {
  if (!column_name %in% colnames(data)) {
    stop("Column not found in the dataframe.")
  }
  
  method <- match.arg(method)
  
  # Ensure the column has numeric data and remove missing values
  if (!is.numeric(data[[column_name]])) {
    stop("Selected column must be numeric.")
  }
  
  # Remove rows where the column has NA values before detecting outliers
  data <- data[!is.na(data[[column_name]]), ]
  
  if (method == "Zscore") {
    z_scores <- scale(data[[column_name]], center = TRUE, scale = TRUE)
    outliers <- abs(z_scores) > z_threshold
  } else if (method == "IQR") {
    Q1 <- quantile(data[[column_name]], 0.25, na.rm = TRUE)
    Q3 <- quantile(data[[column_name]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    outliers <- data[[column_name]] < lower_bound | data[[column_name]] > upper_bound
  }
  
  # Remove outliers while keeping structure intact
  if (method != "Zscore" & method != "IQR") {
    data_cleaned <- data
  } else {
    data_cleaned <- data[!outliers, , drop = FALSE] 
  }
  
  
  return(data_cleaned)
}


############################################################################################################################
#' Single site Frequency Estimation
#'
#' This function performs flood frequency estimation using extreme value analysis.
#'
#' @param dataframe A data frame containing the flood data.
#' @param fit_data A vector of flood data to fit the extreme value distribution.
#' @param cov_data A vector of covariate data (e.g., precipitation).
#' @param Type The type of extreme value distribution to fit (currently supports GEV, and Gumbel). Default is GEV
#' @param Method The method for likelihood estimation (e.g., MLE, GMLE, and Bayesian). Default is MLE
#' @param NS_logic A logical indicating whether to consider stationary (=0) or non-stationarity (=1) (default = 0).
#' @param RP_list A vector of return periods for which to estimate flood frequencies. (default is 25, 50, and 100)
#' @param iter The number of iterations for likelihood estimation (default = 501). This is only used for Bayesian paarmeter estimation methodology
#' @param sig_level The significance level for hypothesis testing.
#'
#' @return A list containing:
#'   \item{sample_len}{Sample length of the at-site annual maxima time series}
#'   \item{mkt}{The Mann-Kendall trend test results.}. The value can be retrieved from the output as ```as.numeric(Output[[1]][[2]][2])```
#'   \item{aic}{The Akaike information criterion (AIC) values for model comparison.} returns NULL for stationary model
#'   \item{parn}{A matrix of estimated parameters (location, scale, shape).}
#'   \item{rln}{A matrix of estimated return levels.}
#'   \item{meanv}{mean value of the time series of Annual Maxima}
#'
#' @details This function assumes that the input data are annual maximum flood values.
#' The function performs a Mann-Kendall trend test to determine if the data exhibit significant trends.
#' Based on the trend test results, it either fits a stationary or non-stationary extreme value distribution using maximum likelihood estimation.
#' The function then estimates flood frequencies for the specified return periods.
#'
#' @references
#'   Katz, R. W., & Brown, B. G. (1992). Extreme events in a changing climate: Variability is more important than averages. Climatic Change, 21(3), 289-302.
#'   Coles, S. (2001). An introduction to statistical modeling of extreme values. Springer.
############################################################################################################################


# Define the function
single_site_freq_estimation <- function(cod_id,
                                        dataframe,
                                        fit_data,
                                        cov_data='None',
                                        Type='GEV',
                                        Method='MLE',
                                        NS_logic=0,
                                        RP_list=c(25,50,100),
                                        iter=1001,
                                        sig_level=0.05) {
  
  meanv = mean(fit_data)
  sample_len = length(fit_data)
  mkt = MannKendall(fit_data)
  pval_trend = as.numeric(mkt[2])
  mktdf <- data.frame(tau = as.numeric(mkt[1]), pvalue = pval_trend)
  
  if(pval_trend <= sig_level && NS_logic == 1) {
    print('Your data have significant trend: Assuming Non-Stationarity')
    
    # Model fitting
    m0 <- fevd(fit_data, data=dataframe, location.fun = ~1, scale.fun = ~1, shape.fun = ~1,
               type = Type, method = Method, use.phi = TRUE, iter=iter)
    
    if(is.character(cov_data)) {
      cov_data = seq_len(sample_len)
    }
    
    m1 <- fevd(fit_data, data=dataframe, location.fun = ~cov_data, scale.fun = ~1, shape.fun = ~1,
               type = Type, method = Method, use.phi = TRUE, iter=iter)
    
    m2 <- fevd(fit_data, data=dataframe, location.fun = ~cov_data, scale.fun = ~cov_data, shape.fun = ~1,
               type = Type, method = Method, use.phi = TRUE, iter=iter)
    
    # Model comparison
    p01 = lr.test(m0, m1, alpha = sig_level)
    p02 = lr.test(m0, m2, alpha = sig_level)
    p12 = lr.test(m1, m2, alpha = sig_level)
    lrt = list(p01=p01$p.value, p02=p02$p.value, p12=p12$p.value)
    
    # AIC comparison
    if(Method == 'Bayesian') {
      aic = c(summary(m0)$DIC, summary(m1)$DIC, summary(m2)$DIC)
    } else {
      aic = c(summary(m0)$AIC, summary(m1)$AIC, summary(m2)$AIC)
    }
    
    # Select best model
    best_model_idx = which.min(aic)
    models = list(m0, m1, m2)
    best_model = models[[best_model_idx]]
    
    # Get parameters and return levels
    par = ci(best_model, type="parameter", alpha=sig_level, return.period=RP_list)
    rl = ci(best_model, type="return.level", alpha=sig_level, return.period=RP_list)
    
    if(best_model_idx == 1) {
      model_type = "Stationary"
      loc = par[1,2]
      scale = par[2,2]
      shape = if(Type == "GEV") par[3,2] else 1e-8
    } else if(best_model_idx == 2) {
      model_type = "Non-Stationary in location"
      loc = max(par[1,2] + par[2,2]*cov_data)
      scale = par[3,2]
      shape = if(Type == "GEV") par[4,2] else 1e-8
    } else {
      model_type = "Non-Stationary in location and scale"
      mu = par[1,2] + par[2,2]*cov_data
      loc = max(mu)
      sigma = exp(par[3,2] + par[4,2]*cov_data)
      scale = sigma[which.max(mu)]
      shape = if(Type == "GEV") par[5,2] else 1e-8
    }
    
  } else {
    print('Assuming Stationarity')
    print(fit_data)
    m0 <- fevd(fit_data, data=dataframe, location.fun = ~1, scale.fun = ~1, shape.fun = ~1,
               type = Type, method = Method, use.phi = TRUE, iter=iter)
    print(summary(m0))
    par = ci(m0, type="parameter", alpha=sig_level, return.period=RP_list)
    rl = ci(m0, type="return.level", alpha=sig_level, return.period=RP_list)
    
    loc = par[1,2]
    scale = par[2,2]
    shape = if(Type == "GEV") par[3,2] else 1e-8
    model_type = "Stationary"
    
    if(Method == 'Bayesian') {
      aic = c(summary(m0)$DIC, NA, NA)
    } else {
      aic = c(summary(m0)$AIC, NA, NA)
    }
    lrt = list(p01=NA, p02=NA, p12=NA)
  }
  
  parn = c(loc=loc, scale=scale, shape=shape)
  # Check if rl is a vector or a matrix/data frame
  if (is.null(dim(rl))) {
    # It's a vector (has length but no dimensions)
    rln = rl[1:3]
  } else {
    # It's a matrix, data frame, or array (has dimensions)
    rln = rl[,1:3]
  }
  
  return(list(
    Sample_length = sample_len,
    mktrend = mktdf,
    AIC = aic,
    lrt = lrt,
    Parameters = parn,
    Index = meanv,
    Best_model = model_type,
    Return_values = rln,
    cod_id = cod_id
  ))
}




#######################################################################################################
# Function: regional_growth_curve_estimation
# Purpose: Estimates the growth curve for a regional phenomenon using the GEV distribution.
# Inputs:
#   - loc_value: Location parameter of the GEV distribution.
#   - scale_value: Scale parameter of the GEV distribution.
#   - shape_value: Shape parameter of the GEV distribution.
#   - return_periods: Vector of return periods for which to estimate growth curve values.
#   - Method: Estimation method ('MLE', 'Bayesian', or 'GMLE').
#   - sig_level: Significance level for the confidence interval.
#   - n_boot: Number of bootstrap resamples for estimating confidence intervals.
# Output:
#   - A data frame with columns:
#     - Return_Period: Specified return periods.
#     - Lower_Quantile: Lower bound of the confidence interval.
#     - Estimate: Estimated growth curve value.
#     - Upper_Quantile: Upper bound of the confidence interval.
########################################################################################################
regional_growth_curve_estimation <- function(loc_value, scale_value, shape_value, return_periods, Method='MLE', sig_level=0.05, n_boot = 1000) {
  
  # Check estimation method
  if (!(Method %in% c("MLE", "Bayesian", "GMLE"))) {
    stop("Method must be one of 'MLE', 'Bayesian', or 'GMLE'")
  }
  
  # Initialize data frame to store results
  results <- data.frame(Return_Period = numeric(length(return_periods)),
                        Lower_Quantile = numeric(length(return_periods)),
                        Estimate = numeric(length(return_periods)),
                        Upper_Quantile = numeric(length(return_periods)))
  
  # Loop through return periods
  for (i in seq_along(return_periods)) {
    rp <- return_periods[i]
    prob <- 1 - (1 / rp)
    # Calculate quantile value
    quantile_value <- qgev(prob, loc = loc_value, scale = scale_value, shape = shape_value, lower.tail = TRUE)
    print(quantile_value)
    # Bootstrap resampling for confidence interval
    set.seed(123)  # For reproducibility
    boot_quantiles <- replicate(n_boot, {
      gev_fit = fevd(x = rgev(100, loc = loc_value, scale = scale_value, shape = shape_value),
                     threshold.fun = ~1, location.fun = ~1,scale.fun = ~1, shape.fun = ~1,type = "GEV", method = Method,initial = NULL,iter=1001)
      location_v=summary(gev_fit)$par[1];scale_v=summary(gev_fit)$par[2];shape_v=summary(gev_fit)$par[3]
      qgev_value=qgev(prob, loc = location_v, scale = scale_v, shape = shape_v, lower.tail = TRUE)
      return(qgev_value)
    })
    
    ci_lower <- quantile(boot_quantiles, sig_level/2)
    ci_upper <- quantile(boot_quantiles, 1-sig_level/2)
    
    # Store results
    results[i, ] <- c(rp, ci_lower, quantile_value, ci_upper)
  }
  
  colnames(results) <- c("Return_Period", "Lower_Quantile", "Estimate", "Upper_Quantile")
  
  return(results)
}


#####################################################################################################################################################################################
# 
# Purpose:
# The regional_frequency_estimation function is designed to perform regional frequency analysis (RFA) on hydrological or meteorological data. It involves several steps:

# Data Input:
# Reads time series data from CSV files.
# Reads spatial information from a shapefile.

# Data Quality Control:
# Detects and removes outliers using specified methods (z-score or IQR).

# At-Site Frequency Analysis:
# Fits a specified probability distribution (e.g., GEV, Gumbel, etc.) to the data using maximum likelihood estimation (MLE), generalized maximum likelihood estimation (GMLE), or Bayesian methods.
# Estimates parameters and calculates return levels for specified return periods.
# Performs statistical tests to assess trend and goodness-of-fit.

# Homogeneous Region Identification:
# Identifies homogeneous regions based on statistical similarity of sites using homogeneity tests (e.g., Anderson-Darling test).

# Regional Frequency Analysis:
# Pools data from homogeneous regions to estimate regional frequency distributions.
# Calculates regional return levels for ungauged sites.

# Output Generation:
# Generates various outputs, including,
# At-site frequency analysis results (e.g., parameter estimates, return levels, goodness-of-fit statistics)
# Homogeneous region assignments
# Regional frequency curves
# Return levels for gauged and ungauged sites

# Input Parameters:
# ts_dir_path: Path to the directory containing time series data in CSV format.
# data_polygon_shapefile_path: Path to the shapefile containing spatial information of the sites.
# fit_data_column_Name: Name of the column in the CSV files containing the data to be analyzed.
# cov_data_column_name: Name of the column in the CSV files containing the covariate data (optional).
# dist_type_df_path: Path to the CSV file containing the distribution type for each site which the user specifies after testing the gof using the gof_test() function.
# save_output_dir_path: Path to the directory where output files will be saved.
# Method: Estimation method ('MLE', 'GMLE', or 'Bayesian').
# cod_ID: Name of the column in the shapefile that identifies the site code.
# Hom_Test: Homogeneity test ('AD' for Anderson-Darling or 'HW' for Höglund).
# outlier_meth: Outlier detection method ('z-score' or 'IQR').
# z_threshold: Threshold for z-score or IQR-based outlier detection.
# NS_logic: Logical flag indicating whether to use non-stationary models.
# RP_list: List of return periods for which to calculate return levels.
# iter: Number of iterations for Bayesian estimation.
# sig_level: Significance level for statistical tests.
# n_boot: Number of bootstrap samples for uncertainty analysis.

# Important Notes and Error Handling:
# Homogeneity Test: The Hom_Test parameter must be either 'AD' or 'HW'.
# Return Period List: The length of the RP_list should not exceed 10.
# Non-stationary Modeling: If NS_logic is 0, the cov_data_column_name must be 'None'. If NS_logic is 1, the cov_data_column_name must be provided.
# Outlier Detection: The outlier_meth and z_threshold parameters must be consistent. If outlier_meth is 'z-score', z_threshold must be provided. If outlier_meth is 'IQR', z_threshold should be 'None'.
# Distribution Type and Estimation Method: The distribution type specified in dist_type_df must be compatible with the chosen estimation method. Bayesian estimation is not supported for the Gumbel distribution.
# Data Consistency: Ensure that the site names in the shapefile and CSV files match and are consistent.
# Data Quality: The input data should be of high quality and free from errors.
# Statistical Assumptions: The assumptions of the statistical methods used should be met.
# Model Selection: The best-fitting model should be selected based on statistical criteria and diagnostic plots.
# Uncertainty Analysis: Uncertainty analysis should be conducted to quantify the uncertainty associated with the estimated return levels.
# Spatial Considerations: The spatial distribution of sites and the underlying hydrological processes should be considered when performing regional frequency analysis.
# 
#####################################################################################################################################################################################
main_regional_freq_analysis <- function(ts_dir_path,
                                        data_polygon_csv_path,  # Changed from shapefile to CSV
                                        fit_data_column_Name,
                                        save_stats_output_dir_path,
                                        save_output_dir_path,
                                        var,
                                        roi,
                                        roi_df_path,
                                        Type_of_dist,
                                        roi_hom_df_path,
                                        gst_list_filenames,
                                        gauging_st_names,
                                        outlier_meth,
                                        z_threshold,
                                        cov_data_column_name = 'Covar',
                                        Method = 'MLE',
                                        NS_logic = 0,
                                        RP_list = c(25, 50, 100),
                                        iter = 1001,
                                        sig_level = 0.05,
                                        n_boot = 501) {
  
  # Read spatial data as CSV instead of shapefile
  geodata <- read.csv(data_polygon_csv_path)
  geodata$Ind_val <- NA  # Initialize the column with NA values
  
  # Initialize output objects
  wb <- createWorkbook()
  site_final <- data.frame()
  FF <- list()
  f=0
  for(filename in gst_list_filenames) {  
    tryCatch({
      f=f+1
      site_name <- sub(paste0("^(full|ams)_", var, "_series_(.*)\\.csv$"), "\\2", filename)
      print(paste0('site name=', site_name))
      print(paste0('file name=', filename))
      if (grepl(paste0("^full_", var, "_series_"), filename)) {
        ts_type <- 'full'
      } else if (grepl(paste0("^ams_", var, "_series_"), filename)) {
        ts_type <- 'ams'
      }
      
      
      # Add proper file path construction
      file_path <- file.path(ts_dir_path, filename)  # Add this line
      
      if (!file.exists(file_path)) {
        warning(paste("File not found:", file_path))
        next
      }
    })
    
    if (ts_type == 'full') {
      df_raws <- read.csv(file_path)
      # Remove columns that contain only NA values
      df_raw <- df_raws[, colSums(is.na(df_raws)) < nrow(df_raws)]
      
      # Find the rows with the maximum variable = ('PI' or 'Flow') for each year
      if (fit_data_column_Name == 'PI') {
        df <- df_raw %>%
          mutate(across(c(PI, Year), as.numeric)) %>%  # Convert to numeric
          filter(!is.na(PI) & !is.na(Year)) %>%  # Remove NAs
          group_by(Year) %>%
          slice_max(PI, n = 1, with_ties = FALSE) %>%  # Keep only the unique max Flow per Year
          ungroup() %>%
          filter(PI != 0)  # Remove rows where Flow is 0
      } else {
        df <- df_raw %>%
          mutate(across(c(Flow, Year), as.numeric)) %>%  # Convert to numeric
          filter(!is.na(Flow) & !is.na(Year)) %>%  # Remove NAs
          group_by(Year) %>%
          slice_max(Flow, n = 1, with_ties = FALSE) %>%  # Keep only the unique max Flow per Year
          ungroup() %>%
          filter(Flow != 0)  # Remove rows where Flow is 0
        df$Flow = df$Flow/df$Area_km2
      }
    } else {
      df_raw <- read.csv(file_path)
      # Remove columns that contain only NA values
      df <- df_raw[, colSums(is.na(df_raw)) < nrow(df_raw)]
      # Remove rows with any missing values
      df <- na.omit(df)
      
      # Check if 'PI' or 'Flow' column exists and remove zero-value rows
      if ('PI' %in% colnames(df)) {
        df <- df[df$PI != 0, ]
      } else if ('Flow' %in% colnames(df)) {
        df$Flow = df$Flow/df$Area_km2
        df <- df[df$Flow != 0, ]
      }
    }
    
    
    # print(df)
    
    # Run function with specified method, column name, and optional Zscore threshold
    dataframe <- detect_and_remove_outliers(df, column_name = fit_data_column_Name, method = outlier_meth, z_threshold = z_threshold)
    
    fit_data = dataframe[[fit_data_column_Name]]
    
    if(NS_logic==1){
      cov_df = dataframe[[cov_data_column_name]]
    }else{
      cov_df='None'
    }
    type_f = Type_of_dist
    
    cod_id = geodata[which(geodata$GWS_ID == site_name),]$Point_ID
    FF[[f]]=single_site_freq_estimation(
      cod_id = cod_id,
      dataframe=dataframe,
      fit_data=fit_data,
      cov_data=cov_df,
      Type=Type_of_dist,
      Method=Method,
      NS_logic=NS_logic,
      RP_list=RP_list,
      iter=iter,
      sig_level=sig_level
    )
    ## Creating a dataframe for adding results of at-site frequency analysis
    site_df = data.frame(cod=cod_id,
                         site=site_name,sample_len=FF[[f]][[1]],  ### PATH TO FILE, AND SAMPLE LENGTH
                         mk_tau = FF[[f]][[2]][[1]],trend_pvalue=FF[[f]][[2]][[2]],   ### MK TREND TEST STATISTICS
                         dist = type_f, Method = Method,  # Type of distribution fitted and method of parameter estimation
                         lrt_0_1 = FF[[f]][[4]][[1]],lrt_0_1 = FF[[f]][[4]][[2]],lrt_0_1 = FF[[f]][[4]][[3]], #### Lilelihood ratio test for best model selection
                         aic_dic_m0 = FF[[f]][[3]][[1]],aic_dic_m1 = FF[[f]][[3]][[2]],aic_dic_m2 = FF[[f]][[3]][[3]],   ### AIC or DIC values
                         best_model = FF[[f]][[7]][[1]], ### Best model selected based on AIC
                         loc = FF[[f]][[5]][[1]],scale = FF[[f]][[5]][[2]],shape = FF[[f]][[5]][[3]],   ##### dist parameters
                         index_flood = FF[[f]][[6]][[1]] ### Index flood
    )
    site_final = rbind(site_final,site_df)
    
    ## Creating dataframe for adding return values (with CI) for speficied return periods
    n<-length(FF[[f]][[8]])/3
    df1 <- data.frame(
      FF[[f]][[8]]
    )
    # Create a new dataframe dff
    dff <- data.frame(
      site = rep(site_name, nrow(df1)),
      return_period = rownames(df1),
      df1
    )
    rownames(dff) <- NULL
    
    ## Writing the return values for each site into a excel sheet with each sheet named after the site name
    addWorksheet(wb, site_name)
    ## Write data frames to the corresponding sheets
    writeData(wb, site_name, dff)
    row_index <- which(geodata$GWS_ID == site_name)
    
    geodata$Ind_val[row_index] <- as.numeric(site_final$index_flood[which(site_final$site==site_name)])
    
  }
  
  if (var =='precip'){
    
    loc_list = site_final$loc
    scale_list = site_final$scale
    shape_list = site_final$shape
    shape_list[shape_list == 0] <- NaN # this is important before averaging, if some of the sites are fit to Gumbel dist.
    sample_len_list = site_final$sample_len
    weighted_avg_location <- weighted.mean(loc_list, sample_len_list)
    weighted_avg_scale <- weighted.mean(scale_list, sample_len_list)
    weighted_avg_shape <- weighted.mean(shape_list, sample_len_list, na.rm = TRUE)
    
    growth_curve =regional_growth_curve_estimation(weighted_avg_location,
                                                   weighted_avg_scale,
                                                   weighted_avg_shape,
                                                   return_periods=RP_list,
                                                   Method=Method, sig_level=sig_level, n_boot = n_boot)
    
    for (gi in 1:length(RP_list)) {
      # Generate the column names based on the Return Period
      col_names <- c(paste0('PI', RP_list[gi], 'yrL'), 
                     paste0('PI', RP_list[gi], 'yrE'), 
                     paste0('PI', RP_list[gi], 'yrU'))
      
      # Extract the corresponding values from growth_curve for the given return period
      values <- as.numeric(growth_curve[gi, 2:4])  # Lower_Quantile, Estimate, Upper_Quantile
      
      # Assign the values to all rows of the respective columns
      geodata[, col_names] <- matrix(rep(values, nrow(geodata)), ncol = 3, byrow = TRUE)
    }
    write.csv(geodata, paste0(save_output_dir_path,"/PIDF_cmperhr_per_watershed_UTM_reprojected.csv"), row.names = FALSE)
    ## Save the workbook with all site's return values
    saveWorkbook(wb, paste0(save_stats_output_dir_path,'/gauged_site_specific_return_values.xlsx'), overwrite = TRUE)
    ## Save the site specific statistics/results of at-site frequency analysis
    write.csv(site_final, file = paste0(save_stats_output_dir_path,'/gauged_site_specific_stats.csv'), row.names = FALSE)
    
  }
  
  if(roi == 1 & var == 'stream' & length(gauging_st_names) >1 ){
    
    
    roi_hom_gauged_file <- file.path(roi_hom_df_path, 'roi_hom_gauged.csv')
    
    if (file.exists(roi_hom_gauged_file)) {
      g_hom_df <- read.csv(roi_hom_gauged_file)
    } else {
      print(paste("File does not exist:", roi_hom_gauged_file))
    }
    
    col_gauged = colnames(g_hom_df)
    for (colg in col_gauged){
      homg_val <- as.list(na.omit(g_hom_df[[colg]]))
      c_values <- which(sapply(FF, function(x) x$cod_id) %in% homg_val)
      
      loc_list <- sapply(FF[c_values], function(x) x$Parameters["loc"])
      scale_list = sapply(FF[c_values], function(x) x$Parameters["scale"])
      shape_list = sapply(FF[c_values], function(x) x$Parameters["shape"])
      shape_list[shape_list == 0] <- NaN # this is important before averaging, if some of the sites are fit to Gumbel dist.
      sample_len_list <- sapply(FF[c_values], function(x) x$Sample_length)
      weighted_avg_location <- weighted.mean(loc_list, sample_len_list)
      weighted_avg_scale <- weighted.mean(scale_list, sample_len_list)
      weighted_avg_shape <- weighted.mean(shape_list, sample_len_list, na.rm = TRUE)
      
      growth_curve =regional_growth_curve_estimation(weighted_avg_location,
                                                     weighted_avg_scale,
                                                     weighted_avg_shape,
                                                     return_periods=RP_list,
                                                     Method=Method, sig_level=sig_level, n_boot = n_boot)
      gauged_point_id <- as.integer(gsub("^X", "", colg))
      gauged_in <- which(geodata$Point_ID %in% gauged_point_id)
      
      for (gi in 1:length(RP_list)) {
        # Generate the column names based on the Return Period
        col_names <- c(paste0('RF', RP_list[gi], 'yrL'), 
                       paste0('RF', RP_list[gi], 'yrE'), 
                       paste0('RF', RP_list[gi], 'yrU'))
        
        # Extract the corresponding values from growth_curve for the given return period
        values <- as.numeric(growth_curve[gi, 2:4])*as.numeric(geodata$Ind_val[gauged_in])*as.numeric(geodata$area_ha[gauged_in]/100)  # Lower_Quantile, Estimate, Upper_Quantile
        
        # Assign the values to all rows of the respective columns
        geodata[gauged_in, col_names] <- matrix(rep(values, 1), ncol = 3, byrow = TRUE)
      }
    }
    
    roi_hom_file <- file.path(roi_hom_df_path, 'roi_hom.csv')
    
    if (file.exists(roi_hom_file)) {
      hom_df <- read.csv(roi_hom_file)
    } else {
      print(paste("File does not exist:", roi_hom_file))
    }
    
    # Function to check unique columns based on un-ordered values and assign unique IDs in the last row
    assign_column_ids <- function(data) {
      # Convert each column to a sorted character string (ignoring NA values) for comparison
      column_patterns <- sapply(data, function(col) paste(sort(na.omit(col)), collapse = "-"))
      # Find unique patterns and assign an ID for each unique column pattern
      unique_patterns <- unique(column_patterns)
      pattern_ids <- paste0("hom_reg", match(column_patterns, unique_patterns))
      # Create a new row with these unique IDs
      new_row <- pattern_ids
      # Add the new row to the data frame
      data <- rbind(data, new_row)
      # Return the modified data frame
      return(data)
    }
    
    # Access dataframe by ID
    hom_df_id <- assign_column_ids(hom_df)
    
    # Function to get unique columns based on the last row IDs
    get_unique_columns <- function(df_with_ids) {
      # Extract the unique ID row (last row) and transpose for easier manipulation
      id_row <- as.character(df_with_ids[nrow(df_with_ids), ])
      # Identify columns with unique ID values
      unique_ids <- unique(id_row)
      unique_columns <- df_with_ids[, !duplicated(id_row), drop = FALSE] # Use drop = FALSE to keep as dataframe
      # Set the column names of the unique columns to match the unique ID values
      colnames(unique_columns) <- unique_ids
      # Remove the last row (ID row) from the output
      unique_columns <- unique_columns[-nrow(unique_columns), , drop = FALSE]
      return(unique_columns)
    }
    
    # Get the unique columns as a new dataframe
    unique_cods <- get_unique_columns(hom_df_id)
    hom_reg = names(unique_cods)
    print(paste0('number of homogeneous regions identified: ',length(hom_reg)))
    list_gc = vector("list", length(hom_reg))
    for(gc in 1:length(hom_reg)){
      hom_cod = as.numeric(na.omit(unique_cods[[hom_reg[gc]]]))
      indices <- which(site_final$cod %in% hom_cod)
      loc_list = site_final$loc[indices]
      scale_list = site_final$scale[indices]
      shape_list = site_final$shape[indices]
      shape_list[shape_list == 0] <- NaN # this is important before averaging, if some of the sites are fit to Gumbel dist.
      sample_len_list = site_final$sample_len[indices]
      weighted_avg_location <- weighted.mean(loc_list, sample_len_list)
      weighted_avg_scale <- weighted.mean(scale_list, sample_len_list)
      weighted_avg_shape <- weighted.mean(shape_list, sample_len_list, na.rm = TRUE)
      print(paste0('Status: deriving growth curve for ',hom_reg[gc]))
      list_gc[[gc]]<- list(region_id = hom_reg[gc],
                           growth_curve =regional_growth_curve_estimation(weighted_avg_location,
                                                                          weighted_avg_scale,
                                                                          weighted_avg_shape,
                                                                          return_periods=RP_list,
                                                                          Method=Method, sig_level=sig_level, n_boot = n_boot))
      
    }
    print("Growth curve estimated")
    ## Calculating the return values for the ungauged catchment from the growth curve and the index flood of the ROI
    roi_index_file <- file.path(roi_df_path, 'roi_index.csv')
    
    if (file.exists(roi_index_file)) {
      ung_roi_df <- read.csv(roi_index_file)
    } else {
      print(paste("File does not exist:", roi_index_file))
    }
    
    ung_roi_df = rbind(ung_roi_df,hom_df_id[nrow(hom_df_id),])
    # Extract non-matching GWS_IDs
    non_matching_gsts <- na.omit(geodata$GWS_ID[!is.na(geodata$GWS_ID) & !(geodata$GWS_ID %in% gauging_st_names)])
    print(non_matching_gsts)
    # Filter the data to include both conditions
    ung_df <- geodata[geodata$Flag_Gst == 0 | geodata$GWS_ID %in% non_matching_gsts, ]
    for(un in 1:nrow(ung_df)){
      ung_cod = ung_df$Point_ID[un]
      ung_cod_ID=paste0('X',ung_cod)
      ung_hom_reg_id = ung_roi_df[[ung_cod_ID]][nrow(ung_roi_df)]
      # Define the target homogeneous region ID you want to retrieve
      target_region_id <- ung_hom_reg_id  # Replace with the desired region ID
      
      # Find the growth curve for the specified homogeneous region ID
      growth_curve <- NULL
      for (entry in list_gc) {
        if (entry$region_id == target_region_id) {
          growth_curve <- entry$growth_curve
          break  # Stop the loop once the target region is found
        }
      }
      
      roi_cod = as.numeric(na.omit(ung_roi_df[[ung_cod_ID]]))[1]
      ung_index = site_final$index_flood[which(site_final$cod %in% roi_cod)]
      ind = which(geodata$Point_ID == ung_cod)
      geodata$Ind_val[ind] <- as.numeric(ung_index)
      ung_RV= growth_curve[,2:4]*as.numeric(ung_index)*as.numeric(geodata$area_ha[ind]/100)
      
      for (gi in 1:length(RP_list)) {
        # Generate column names dynamically for RF values
        col_names <- c(paste0('RF', RP_list[gi], 'yrL'), 
                       paste0('RF', RP_list[gi], 'yrE'), 
                       paste0('RF', RP_list[gi], 'yrU'))
        
        # Extract the corresponding values from ung_RV for the given return period
        values <- as.numeric(ung_RV[gi, 1:3])  # Lower_Quantile, Estimate, Upper_Quantile
        
        # Find the rows where Point_ID matches ung_cod
        rows_to_update <- which(geodata$Point_ID == ung_cod)
        
        # Assign values to the selected rows
        geodata[rows_to_update, col_names] <- matrix(rep(values, length(rows_to_update)), 
                                                     ncol = 3, byrow = TRUE)
      }
      
    }
    # Create a new workbook
    wb_gc <- createWorkbook()
    
    # Loop through each item in list_gc to add sheets
    for (entry in list_gc) {
      # Extract region ID and growth curve data
      region_id <- entry$region_id
      growth_curve <- entry$growth_curve
      
      # Add a new sheet named after the region_id
      addWorksheet(wb_gc, sheetName = region_id)
      
      # Write the growth curve data to the sheet
      writeData(wb_gc, sheet = region_id, growth_curve)
    }
    
    # Saving the homogeneous region : gauged cod and region ID for each ungauged catchment
    write.csv(hom_df_id, file = paste0(save_stats_output_dir_path,'/ungauged_site_hom_region_ID_with_ROI.csv'), row.names = FALSE)
    print("saved the hom region gauged cod and region ID for each ungauged catchments as ungauged_site_hom_region_ID_with_ROI.csv")
    # Saving the workbook as an Excel file
    saveWorkbook(wb_gc, paste0(save_stats_output_dir_path,"/growth_curves_by_hom_region.xlsx"), overwrite = TRUE)
    print("Excel file 'growth_curves_by_hom_region.xlsx' has been saved with each growth curve quantile in a separate sheet.")
    # Saving the final results with Return values in a geodataframe
    write.csv(geodata, paste0(save_output_dir_path,"/RFA_results_of_return_values.csv"), row.names = FALSE)
    print("Saving the final results with Return values as RFA_results_of_return_values.csv")
    
  }else{
    if (var =='stream' & length(gst_list_filenames)==1){
      
      loc_list = site_final$loc
      scale_list = site_final$scale
      shape_list = site_final$shape
      shape_list[shape_list == 0] <- NaN # this is important before averaging, if some of the sites are fit to Gumbel dist.
      sample_len_list = site_final$sample_len
      weighted_avg_location <- weighted.mean(loc_list, sample_len_list)
      weighted_avg_scale <- weighted.mean(scale_list, sample_len_list)
      weighted_avg_shape <- weighted.mean(shape_list, sample_len_list, na.rm = TRUE)
      
      growth_curve =regional_growth_curve_estimation(weighted_avg_location,
                                                     weighted_avg_scale,
                                                     weighted_avg_shape,
                                                     return_periods=RP_list,
                                                     Method=Method, sig_level=sig_level, n_boot = n_boot)
      
      for (gi in 1:length(RP_list)) {
        # Generate the column names based on the Return Period
        col_names <- c(paste0('RF', RP_list[gi], 'yrL'), 
                       paste0('RF', RP_list[gi], 'yrE'), 
                       paste0('RF', RP_list[gi], 'yrU'))
        
        # Extract the corresponding values from growth_curve for the given return period
        gc_values <- as.numeric(growth_curve[gi, 2:4])  # Lower_Quantile, Estimate, Upper_Quantile
        index_val = site_final$index_flood
        area_val = as.numeric(geodata$area_ha[which(geodata$GWS_ID ==gauging_st_names)]/100) # in km2
        # Assign the values to all rows of the respective columns
        geodata[, col_names] <- matrix(rep(gc_values, nrow(geodata)), ncol = 3, byrow = TRUE)*as.numeric(index_val)*area_val
      }
      ## Save the workbook with all site's return values
      saveWorkbook(wb, paste0(save_stats_output_dir_path,'/gauged_site_specific_return_values.xlsx'), overwrite = TRUE)
      ## Save the site specific statistics/results of at-site frequency analysis
      write.csv(site_final, file = paste0(save_stats_output_dir_path,'/gauged_site_specific_stats.csv'), row.names = FALSE)
      
      write.csv(geodata, paste0(save_output_dir_path,"/RFA_results_of_return_values.csv"), row.names = FALSE)
      print("Saving the final results with Return values as RFA_results_of_return_values.csv")
    }
    
  }
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
def main_regional_freq_analysis(ts_dir_path,
                                data_polygon_shapefile_path,  # Still accept shapefile path
                                fit_data_column_Name,
                                save_stats_output_dir_path,
                                save_output_dir_path,
                                var,
                                roi,
                                roi_df_path,
                                Type_of_dist,
                                roi_hom_df_path,
                                gst_list_filenames,
                                gauging_st_names,
                                outlier_meth,
                                z_threshold,
                                cov_data_column_name,
                                Method='MLE',
                                NS_logic=0,
                                RP_list=None,
                                iter=1001,
                                sig_level=0.05,
                                n_boot=501,
                                return_geodata=True):
    try:
        # Convert shapefile to CSV for R processing
        gdf = gpd.read_file(data_polygon_shapefile_path)
        
        # Create temporary CSV file for R
        csv_path = data_polygon_shapefile_path.replace('.shp', '_temp.csv')
        gdf.drop(columns='geometry').to_csv(csv_path, index=False)
        
        # Convert parameters for R
        RP_list_r = "c(25,50,100)" if RP_list is None else "c(" + ",".join(str(x) for x in RP_list) + ")"
        stations_r = "c(" + ", ".join(f"'{st}'" for st in gauging_st_names) + ")" if gauging_st_names else "c()"
        gst_list_filenames_r = "c(" + ", ".join(f"'{st}'" for st in gst_list_filenames) + ")" if gst_list_filenames else "c()"

        # Run R analysis with CSV input
        robjects.r(
            f"""
            main_regional_freq_analysis(
                ts_dir_path = '{ts_dir_path}',
                data_polygon_csv_path = '{csv_path}',  # Use CSV instead
                fit_data_column_Name = '{fit_data_column_Name}',
                save_stats_output_dir_path = '{save_stats_output_dir_path}',
                save_output_dir_path = '{save_output_dir_path}',
                var = '{var}',
                roi = {roi},  
                roi_df_path = '{roi_df_path}',
                Type_of_dist = '{Type_of_dist}',
                roi_hom_df_path = '{roi_hom_df_path}',
                gst_list_filenames = {gst_list_filenames_r},
                gauging_st_names = {stations_r},
                outlier_meth = '{outlier_meth}',
                z_threshold = {z_threshold},
                cov_data_column_name = '{cov_data_column_name}',
                Method = '{Method}',
                NS_logic = {NS_logic},
                RP_list = {RP_list_r},
                iter = {iter},
                sig_level = {sig_level},
                n_boot = {n_boot}
            )
            """
        )
        
        # Clean up temporary CSV
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        if return_geodata:
            # Read the CSV results and merge back with geometry
            if var == 'precip':
                result_csv = f"{save_output_dir_path}/PIDF_cmperhr_per_watershed_UTM_reprojected.csv"
            else:
                result_csv = f"{save_output_dir_path}/RFA_results_of_return_values.csv"
            
            # Read results and merge with original geometry
            result_df = pd.read_csv(result_csv)
            
            # Merge results with original geodataframe geometry
            # Assuming there's a common ID column to join on
            # Merge results with original geodataframe geometry
            common_cols = list(set(gdf.columns).intersection(set(result_df.columns)))
            if 'geometry' in common_cols:
                common_cols.remove('geometry')

            if common_cols:
                # Use the first common column as merge key
                merge_key = common_cols[0]
                result_gdf = gdf.merge(result_df, on=merge_key, how='left', suffixes=('', '_r'))
            else:
                # If no common columns, assume same order and concatenate
                result_gdf = gdf.copy()
                for col in result_df.columns:
                    if col not in result_gdf.columns:
                        result_gdf[col] = result_df[col].values
            
            # Save as shapefile using GeoPandas
            if var == 'precip':
                output_shp = f"{save_output_dir_path}/PIDF_cmperhr_per_watershed_UTM_reprojected.shp"
            else:
                output_shp = f"{save_output_dir_path}/RFA_results_of_return_values.shp"
            
            result_gdf.to_file(output_shp)
            return result_gdf
            
        return None

    except Exception as e:
        # Clean up temporary files
        csv_path = data_polygon_shapefile_path.replace('.shp', '_temp.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
        raise Exception(f"Error in Regional Frequency Analysis: {str(e)}")