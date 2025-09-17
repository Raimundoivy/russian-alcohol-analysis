import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pandas import DataFrame, Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper


def plotly_forecast(original_df: DataFrame, forecast_df: DataFrame) -> go.Figure:
    """Generates an interactive forecast plot using Plotly."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[col.title() for col in original_df.columns],
        shared_xaxes=True,
    )
    row, col = 1, 1
    for i, column in enumerate(original_df.columns):
        fig.add_trace(
            go.Scatter(
                x=original_df.index,
                y=original_df[column],
                name="Historical",
                mode="lines",
                legendgroup="Historical",
                showlegend=(i == 0),
                line=dict(color="blue"),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df[column],
                name="Forecast",
                mode="lines",
                legendgroup="Forecast",
                showlegend=(i == 0),
                line=dict(color="red", dash="dash"),
            ),
            row=row,
            col=col,
        )
        col += 1
        if col > 2:
            col = 1
            row += 1

    fig.update_layout(
        title_text="Alcohol Consumption Forecast",
        height=600,
        legend_tracegroupgap=180,
    )
    return fig


def plotly_correlation_heatmap(df: DataFrame) -> go.Figure:
    """Generates an interactive correlation heatmap using Plotly."""
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        # FIX: Changed 'coolwarm' to the valid Plotly colorscale 'RdBu'
        color_continuous_scale="RdBu",
        title="Correlation Matrix of Annual Consumption Changes",
    )
    return fig


def plotly_trend_decomposition(series: Series, series_name: str) -> go.Figure:
    """Generates an interactive trend decomposition plot using Plotly."""
    decomposition = seasonal_decompose(series, model="additive", period=1)
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=["Trend", "Seasonality", "Residual"],
        shared_xaxes=True,
    )
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode="lines", name="Trend"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.seasonal.index, y=decomposition.seasonal, mode="lines", name="Seasonality"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode="markers", name="Residual"),
        row=3,
        col=1,
    )
    fig.update_layout(
        title_text=f"Trend Decomposition for {series_name.title()}",
        height=700,
        showlegend=False,
    )
    return fig


def plotly_backtesting(actual_df: DataFrame, forecast_df: DataFrame) -> go.Figure:
    """Generates an interactive plot comparing backtest forecast vs. actuals."""
    fig = go.Figure()
    for col in actual_df.columns:
        fig.add_trace(
            go.Scatter(
                x=actual_df.index,
                y=actual_df[col],
                name=f"Actual {col.title()}",
                mode="lines+markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df[col],
                name=f"Forecast {col.title()}",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title="Backtesting: Forecast vs. Actuals",
        xaxis_title="Year",
        yaxis_title="Liters per capita",
        legend_title="Beverage",
    )
    return fig


def plotly_what_if_forecast(
    original_forecast: DataFrame, what_if_forecast: DataFrame
) -> go.Figure:
    """Generates an interactive plot comparing the original and what-if forecasts."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[col.title() for col in original_forecast.columns],
        shared_xaxes=True,
    )
    row, col = 1, 1
    for i, column in enumerate(original_forecast.columns):
        fig.add_trace(
            go.Scatter(
                x=original_forecast.index,
                y=original_forecast[column],
                name="Original Forecast",
                mode="lines",
                legendgroup="Original",
                showlegend=(i == 0),
                line=dict(color="blue", dash="dash"),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=what_if_forecast.index,
                y=what_if_forecast[column],
                name="What-If Forecast",
                mode="lines",
                legendgroup="What-If",
                showlegend=(i == 0),
                line=dict(color="green", dash="dot"),
            ),
            row=row,
            col=col,
        )
        col += 1
        if col > 2:
            col = 1
            row += 1

    fig.update_layout(
        title_text="What-If Scenario Forecast Comparison", height=600
    )
    return fig