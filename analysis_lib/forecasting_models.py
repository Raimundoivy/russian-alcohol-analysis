from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VARResults
import pandas as pd
from pandas import DataFrame
from typing import Dict, Tuple, List


def train_var_model(df: DataFrame) -> VARResults:
    """
    Trains a Vector Autoregression (VAR) model on the provided DataFrame.

    Args:
        df (DataFrame): The input DataFrame for training the model.

    Returns:
        VARResults: The fitted VAR model.
    """
    model = VAR(df)
    fitted_model = model.fit(1)
    return fitted_model


def generate_forecast(fitted_model: VARResults, steps: int = 5) -> DataFrame:
    """
    Generates a forecast using a fitted VAR model.

    Args:
        fitted_model (VARResults): The fitted VAR model.
        steps (int, optional): The number of steps to forecast. Defaults to 5.

    Returns:
        DataFrame: A DataFrame with the forecasted values.
    """
    lag_order = fitted_model.k_ar
    forecast_input = fitted_model.model.endog[-lag_order:]
    forecast = fitted_model.forecast(y=forecast_input, steps=steps)
    last_date = fitted_model.model.data.dates[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.DateOffset(years=1), periods=steps, freq="A"
    )
    forecast_df = pd.DataFrame(
        forecast, index=forecast_index, columns=fitted_model.model.endog_names
    )
    return forecast_df


def get_granger_causality_results(
    df: DataFrame, variables: List[str], max_lag: int = 1
) -> Dict[Tuple[str, str], float]:
    """
    Performs Granger causality tests for all pairs of variables.

    Args:
        df (DataFrame): The input DataFrame.
        variables (List[str]): A list of variable names to test.
        max_lag (int, optional): The maximum lag to test for. Defaults to 1.

    Returns:
        Dict[Tuple[str, str], float]: A dictionary with the p-values of the Granger causality tests.
    """
    test_results = {}
    for var1 in variables:
        for var2 in variables:
            if var1 != var2:
                test_result = grangercausalitytests(
                    df[[var2, var1]], maxlag=max_lag, verbose=False
                )
                p_value = test_result[max_lag][0]["ssr_ftest"][1]
                test_results[(var1, var2)] = p_value
    return test_results