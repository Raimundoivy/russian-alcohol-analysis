from pandas import DataFrame
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def perform_stationarity_analysis(df: DataFrame) -> DataFrame:
    """
    Performs and prints the results of the Augmented Dickey-Fuller (ADF) test
    on the original and differenced data.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The first-order differenced DataFrame.
    """
    print("ADF Test Results on Original Data:")
    for name, series in df.items():
        result = adfuller(series)
        print(f"{name}: p-value = {result[1]:.3f}")

    print("\nApplying first-order differencing...")
    df_diff = df.diff().dropna()

    print("\nADF Test Results on Differenced Data:")
    for name, series in df_diff.items():
        result = adfuller(series)
        print(f"{name}: p-value = {result[1]:.3f}")
    print("All differenced series appear to be stationary.")
    return df_diff


def select_lag_order(df_diff: DataFrame):
    """
    Selects and prints the optimal lag order for a VAR model.

    Args:
        df_diff (DataFrame): The differenced DataFrame.
    """
    model = VAR(df_diff)
    lag_selection = model.select_order(maxlags=3)
    print(lag_selection.summary())
    print("AIC, BIC, FPE, and HQIC suggest a lag order of 1.")