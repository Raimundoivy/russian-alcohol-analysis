import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import DataFrame, Series
from statsmodels.tsa.vector_ar.var_model import VARResults
from matplotlib.figure import Figure


def plot_forecast(original_df: DataFrame, forecast_df: DataFrame) -> Figure:
    """
    Plots the original data and the forecasted values for all categories.

    Args:
        original_df (DataFrame): The DataFrame with the original data.
        forecast_df (DataFrame): The DataFrame with the forecasted data.

    Returns:
        Figure: The matplotlib Figure object.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)
    fig.suptitle("Alcohol Consumption Forecast", fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, col in enumerate(original_df.columns):
        if i < len(axes):  # Ensure we don't try to plot more columns than we have subplots
            ax = axes[i]
            original_df[col].plot(ax=ax, label="Historical", legend=True)
            forecast_df[col].plot(ax=ax, label="Forecast", legend=True, linestyle="--")
            ax.set_title(col.replace("_", " ").title())
            ax.set_ylabel("Liters per capita")
            ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_correlation_heatmap(df: DataFrame) -> Figure:
    """
    Plots a correlation heatmap for the DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        Figure: The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix of Alcohol Consumption Types")
    return fig


def plot_residuals(fitted_model: VARResults) -> Figure:
    """
    Plots the residuals of a fitted VAR model.

    Args:
        fitted_model (VARResults): The fitted VAR model.

    Returns:
        Figure: The matplotlib Figure object.
    """
    residuals = fitted_model.resid
    # Ensure there's a subplot for each column
    num_plots = residuals.shape[1]
    fig, axes = plt.subplots(nrows=num_plots, figsize=(10, 2 * num_plots), sharex=True)
    if num_plots == 1:  # Handle case with only one plot
        axes = [axes]
    fig.suptitle("Model Residuals Analysis", fontsize=16)
    for i, col in enumerate(residuals.columns):
        ax = axes[i]
        residuals[col].plot(ax=ax)
        ax.set_title(f"Residuals for {col.title()}")
        ax.axhline(0, color="r", linestyle="--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_trend_decomposition(series: Series, series_name: str) -> Figure:
    """
    Plots the trend, seasonal, and residual components of a time series.

    Args:
        series (Series): The time series to decompose.
        series_name (str): The name of the time series.

    Returns:
        Figure: The matplotlib Figure object.
    """
    decomposition = seasonal_decompose(series, model="additive", period=1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Trend Decomposition for {series_name.title()}", fontsize=16)

    decomposition.trend.plot(ax=ax1, title="Trend")
    decomposition.seasonal.plot(ax=ax2, title="Seasonality")
    decomposition.resid.plot(ax=ax3, title="Residual")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig