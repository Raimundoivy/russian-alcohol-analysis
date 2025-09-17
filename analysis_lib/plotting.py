import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast(original_df, forecast_df):
    """
    Plots the original data and the forecasted values for all categories.

    Args:
        original_df (pd.DataFrame): The original time-series data.
        forecast_df (pd.DataFrame): The forecasted data.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('Alcohol Consumption Forecast', fontsize=16)
    
    for i, col in enumerate(original_df.columns):
        ax = axes[i//2, i%2]
        original_df[col].plot(ax=ax, label='Historical', legend=True)
        forecast_df[col].plot(ax=ax, label='Forecast', legend=True, linestyle='--')
        ax.set_title(col.replace('_', ' ').title())
        ax.set_ylabel('Consumption (Liters per capita)')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for the DataFrame.

    Args:
        df (pd.DataFrame): The data to correlate.

    Returns:
        matplotlib.figure.Figure: The figure object containing the heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Alcohol Consumption Types')
    return fig