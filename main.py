import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from analysis_lib.data_loader import load_and_prepare_data
from analysis_lib.forecasting_models import (
    train_var_model,
    generate_forecast,
    train_varmax_model,
)
from analysis_lib.plotting import (
    plot_correlation_heatmap,
    plot_forecast,
    plot_impulse_response,
)
from analysis_lib.eda import perform_stationarity_analysis, select_lag_order


def main():
    """
    Main function to run the alcohol consumption analysis pipeline.
    """
    # --- 1. Data Loading and Preparation ---
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print("Data loaded successfully.")
    print(df.head())

    # --- 2. Exploratory Data Analysis ---
    print("\nPlotting time series for each beverage...")
    df.plot(
        subplots=True,
        figsize=(15, 10),
        layout=(2, 2),
        title="Per Capita Alcohol Consumption in Russia",
    )
    plt.show()

    print("\nPerforming stationarity analysis...")
    df_diff = perform_stationarity_analysis(df)

    print("\nGenerating correlation heatmap on differenced data...")
    fig_heatmap = plot_correlation_heatmap(df_diff)
    plt.show()

    # --- 3. VAR Model Training and Analysis ---
    print("\nSelecting optimal lag order for VAR model...")
    select_lag_order(df_diff)

    print("\nTraining the VAR model on differenced data...")
    fitted_model = train_var_model(df_diff)
    print(fitted_model.summary())

    print("\nPlotting Impulse Response Functions...")
    fig_irf = plot_impulse_response(fitted_model)
    plt.show()

    # --- 4. Forecasting ---
    print("\nGenerating 5-year forecast...")
    # Note: Forecasting is done on the original model trained on level data
    # for easier interpretation.
    level_model = train_var_model(df)
    forecast_df = generate_forecast(level_model, steps=5)
    print("Forecast generated successfully.")
    print(forecast_df)

    print("\nPlotting the forecast...")
    fig_forecast = plot_forecast(df, forecast_df)
    plt.show()

    # --- 5. VARMAX Model Experiment ---
    print("\nTraining VARMAX model to account for COVID-19...")
    df["covid"] = np.where(df.index.year >= 2020, 1, 0)
    df_diff_covid = df.diff().dropna()
    endog_vars = ["wine", "beer", "vodka", "brandy"]
    exog_vars = ["covid"]
    fitted_varmax_model = train_varmax_model(
        df_diff_covid[endog_vars], df_diff_covid[exog_vars]
    )
    print(fitted_varmax_model.summary())
    print("\nVARMAX model training complete.")


if __name__ == "__main__":
    main()