import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_forecast(original_df, forecast_df):
    """
    Plots the original data and the forecasted values for all categories.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)
    fig.suptitle('Alcohol Consumption Forecast', fontsize=16)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, col in enumerate(original_df.columns):
        if i < len(axes): # Ensure we don't try to plot more columns than we have subplots
            ax = axes[i]
            original_df[col].plot(ax=ax, label='Historical', legend=True)
            forecast_df[col].plot(ax=ax, label='Forecast', legend=True, linestyle='--')
            ax.set_title(col.replace('_', ' ').title())
            ax.set_ylabel('Liters per capita')
            ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for the DataFrame.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Alcohol Consumption Types')
    return fig

def plot_residuals(fitted_model):
    """
    Plots the residuals of a fitted VAR model.
    """
    residuals = fitted_model.resid
    # Ensure there's a subplot for each column
    num_plots = residuals.shape[1]
    fig, axes = plt.subplots(nrows=num_plots, figsize=(10, 2 * num_plots), sharex=True)
    if num_plots == 1: # Handle case with only one plot
        axes = [axes]
    fig.suptitle('Model Residuals Analysis', fontsize=16)
    for i, col in enumerate(residuals.columns):
        ax = axes[i]
        residuals[col].plot(ax=ax)
        ax.set_title(f'Residuals for {col.title()}')
        ax.axhline(0, color='r', linestyle='--')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_trend_decomposition(series, series_name):
    """
    Plots the trend, seasonal, and residual components of a time series.
    """
    decomposition = seasonal_decompose(series, model='additive', period=1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Trend Decomposition for {series_name.title()}', fontsize=16)
    
    decomposition.trend.plot(ax=ax1, title='Trend')
    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
    decomposition.resid.plot(ax=ax3, title='Residual')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig