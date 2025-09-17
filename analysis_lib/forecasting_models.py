from statsmodels.tsa.api import VAR
import pandas as pd

def train_var_model(df):
    """
    Trains a Vector Autoregression (VAR) model on the provided DataFrame.

    Args:
        df (pd.DataFrame): The time-series data.

    Returns:
        statsmodels.tsa.vector_ar.var_model.VARResultsWrapper: The fitted VAR model results.
    """
    model = VAR(df)
    # Using a lag order of 1 as determined in the notebook's analysis
    fitted_model = model.fit(1)
    return fitted_model

def generate_forecast(fitted_model, steps=5):
    """
    Generates a forecast using a fitted VAR model.

    Args:
        fitted_model: The fitted VAR model results object.
        steps (int): The number of future steps to forecast.

    Returns:
        pandas.DataFrame: A DataFrame containing the forecasted values.
    """
    lag_order = fitted_model.k_ar
    
    # Get the last `lag_order` values from the original data to start the forecast
    forecast_input = fitted_model.model.endog[-lag_order:]
    
    # Generate the forecast
    forecast = fitted_model.forecast(y=forecast_input, steps=steps)
    
    # --- FIX: Use model.data.dates to get the index of the original data ---
    # Create a date range for the forecast period
    last_date = fitted_model.model.data.dates[-1]
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1),
                                   periods=steps,
                                   freq='A') # 'A' for Annual frequency
                                   
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=fitted_model.model.endog_names)
    return forecast_df