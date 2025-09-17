from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def train_var_model(df):
    """
    Trains a Vector Autoregression (VAR) model on the provided DataFrame.
    """
    model = VAR(df)
    fitted_model = model.fit(1)
    return fitted_model

def generate_forecast(fitted_model, steps=5):
    """
    Generates a forecast using a fitted VAR model.
    """
    lag_order = fitted_model.k_ar
    forecast_input = fitted_model.model.endog[-lag_order:]
    forecast = fitted_model.forecast(y=forecast_input, steps=steps)
    last_date = fitted_model.model.data.dates[-1]
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=steps, freq='A')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=fitted_model.model.endog_names)
    return forecast_df

def get_granger_causality_results(df, variables, max_lag=1):
    """
    Performs Granger causality tests for all pairs of variables.
    """
    test_results = {}
    for var1 in variables:
        for var2 in variables:
            if var1 != var2:
                test_result = grangercausalitytests(df[[var2, var1]], maxlag=max_lag, verbose=False)
                p_value = test_result[max_lag][0]['ssr_ftest'][1]
                test_results[(var1, var2)] = p_value
    return test_results