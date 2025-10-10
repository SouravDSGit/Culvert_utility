import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, conversion
from functools import wraps
from rpy2.robjects.conversion import localconverter

# First define the R code
robjects.r(
    r"""
    
# --------------------------------- DETECT OUTLIERS --------------------------------------------------------
detect_and_remove_outliers <- function(data, column_name, method = c("Z-score","IQR","None"), z_threshold = 3) {
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

  if (method == "Z-score") {
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
  if (method != "Z-score" & method != "IQR") {
   data_cleaned <- data
  } else {
   data_cleaned <- data[!outliers, , drop = FALSE] 
  }
  

  return(data_cleaned)
}


# -------------------------  PERFORM SINGLE SITE FREQUENCY ANALYSIS --------------------------------------
single_site_freq_estimation <- function(dataframe,
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
    Return_values = rln
  ))
}

# --------------------------------------- Main function -----------------------------
main_single_site_freq_analysis <- function(save_output_dir_path,
                                         ts_dir_path,
                                         gst_list_filenames,
                                         fit_data_column_Name,
                                         outlier_meth,
                                         z_threshold,
                                         var,
                                         gauging_st_names,
                                         Type_of_dist='GEV',
                                         Method='MLE',
                                         NS_logic=0,
                                         cov_data_column_name='Covar',
                                         RP_list=c(25,50,100),
                                         iter=1001,
                                         sig_level=0.05) {
  
  # Initialize output objects
  wb <- createWorkbook()
  site_final <- data.frame()
  FF <- list()
  
  
  for(filename in gst_list_filenames) {  
      tryCatch({
          
          site_name <- sub(paste0("^(full|ams)_", var, "_series_(.*)\\.csv$"), "\\2", filename)

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
              group_by(Year) %>%
              filter(PI == max(PI, na.rm = TRUE)) %>%
              ungroup()
          df <- df[df$PI != 0, ]
      } else {
          df <- df_raw %>%
              group_by(Year) %>%
              filter(Flow == max(Flow, na.rm = TRUE)) %>%
              ungroup()
          df <- df[df$Flow != 0, ]
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
          df <- df[df$Flow != 0, ]
      }
  }


    print(df)
    # Remove outliers if specified
    if(!is.null(outlier_meth)) {
      df <- detect_and_remove_outliers(df, 
                                     column_name=fit_data_column_Name, 
                                     method=outlier_meth, 
                                     z_threshold=z_threshold)
    }
    print('outlier removed')
    # Prepare data for frequency analysis
    
    fit_data <- df[[fit_data_column_Name]]
    cov_data <- if (NS_logic == 1 && cov_data_column_name != 'None') {
        if (cov_data_column_name %in% colnames(df)) {
          df[[cov_data_column_name]]
        } else {
          'None'
        }
      } else {
        'None'
      }
    
    # Perform frequency analysis
    result <- single_site_freq_estimation(
      dataframe=df,
      fit_data=fit_data,
      cov_data=cov_data,
      Type=Type_of_dist,
      Method=Method,
      NS_logic=NS_logic,
      RP_list=RP_list,
      iter=iter,
      sig_level=sig_level
    )
    
    FF[[site_name]] <- result
    
    # Create site statistics
    site_stats <- data.frame(
      site=site_name,
      sample_len=result$Sample_length,
      mk_tau=result$mktrend$tau,
      trend_pvalue=result$mktrend$pvalue,
      dist=Type_of_dist,
      Method=Method,
      best_model=result$Best_model,
      loc=result$Parameters["loc"],
      scale=result$Parameters["scale"],
      shape=result$Parameters["shape"],
      index_flood=result$Index
    )
    
    site_final <- rbind(site_final, site_stats)
    
    # Create return values worksheet
    return_values <- as.data.frame(result$Return_values)
    print(result)
    colnames(return_values) <- c("Lower_CI","Estimate", "Upper_CI")
    return_values$Return_Period <- RP_list
    
    addWorksheet(wb, site_name)
    writeData(wb, site_name, return_values)
  }
  
  # Save results
  saveWorkbook(wb, file.path(save_output_dir_path, 'site_specific_return_values.xlsx'), overwrite=TRUE)
  write.csv(site_final, file.path(save_output_dir_path, 'site_specific_stats.csv'), row.names=FALSE)
  
  return(list(FF=FF, site_stats=site_final))
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
def main_single_site_freq_analysis(save_output_dir_path,
                                 ts_dir_path,
                                 gst_list_filenames,
                                 fit_data_column_Name,
                                 outlier_meth,                                         
                                 z_threshold,
                                 var,
                                 gauging_st_names,
                                 Type_of_dist='GEV',
                                 Method='MLE',
                                 NS_logic=0,
                                 cov_data_column_name='None',
                                 RP_list=None,
                                 iter=1001,
                                 sig_level=0.05):
    """
    Python wrapper for the R frequency analysis function
    """
    try:
        if RP_list is None:
            RP_list = "c(25,50,100)"
        else:
            RP_list = "c(" + ",".join(str(x) for x in RP_list) + ")"
            
        stations_r = "c(" + ", ".join(f"'{st}'" for st in gauging_st_names) + ")"
        gst_list_filenames_r = "c(" + ", ".join(f"'{st}'" for st in gst_list_filenames) + ")"
        
        # Execute the R function
        result = robjects.r(
            f"""
            main_single_site_freq_analysis(
                save_output_dir_path = '{save_output_dir_path}',
                ts_dir_path = '{ts_dir_path}',
                gst_list_filenames = {gst_list_filenames_r},  # Fixed: use the R-formatted version
                fit_data_column_Name = '{fit_data_column_Name}',
                outlier_meth = '{outlier_meth}',                                         
                z_threshold = {z_threshold},
                var = '{var}',
                gauging_st_names = {stations_r},
                Type_of_dist = '{Type_of_dist}',
                Method = '{Method}',
                NS_logic = {NS_logic},
                cov_data_column_name = '{cov_data_column_name}',
                RP_list = {RP_list},
                iter = {iter},
                sig_level = {sig_level}
            )
            """
        )
        
        return result
        
    except Exception as e:
        raise Exception(f"Error in Single Site Frequency Analysis: {str(e)}")