import streamlit as st
import pandas as pd
from analysis_lib.data_loader import load_and_prepare_data
from analysis_lib.forecasting_models import train_var_model, generate_forecast
# Import new plotly plotting library
from analysis_lib.plotly_plotting import (
    plotly_correlation_heatmap,
    plotly_forecast,
    plotly_trend_decomposition,
    plotly_backtesting,
    plotly_what_if_forecast,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Russian Alcohol Consumption Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Caching Functions ---
@st.cache_data
def cached_load_data():
    return load_and_prepare_data()


@st.cache_resource
def cached_train_model(_df):
    return train_var_model(_df)


# --- Main Application ---
st.title("🍾 Russian Alcohol Consumption Analysis Dashboard")
st.markdown(
    "An interactive dashboard for exploring, forecasting, and analyzing alcohol consumption trends in Russia from 1998 to 2023."
)

# --- Load Data ---
try:
    df_full = cached_load_data()
except FileNotFoundError:
    st.error("Error: The dataset file was not found.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
selected_beverages = st.sidebar.multiselect(
    "Select beverages to analyze:",
    options=df_full.columns,
    default=list(df_full.columns),
)

if not selected_beverages:
    st.warning("Please select at least one beverage to continue.")
    st.stop()

df_selection = df_full[selected_beverages]

# --- Main Tabs ---
tabs = st.tabs(
    [
        "📊 Exploratory Data Analysis",
        "📈 Forecasting",
        "🔬 Backtesting",
        "❓ What-If Analysis",
    ]
)

# --- Exploratory Data Analysis Tab ---
with tabs[0]:
    st.header("Exploratory Data Analysis")

    st.subheader("Time Series Data")
    st.line_chart(df_selection)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        if len(df_selection.columns) > 1:
            st.plotly_chart(
                plotly_correlation_heatmap(df_selection.diff().dropna()),
                use_container_width=True,
            )
        else:
            st.info("Select at least two beverages to see their correlation.")

    with col2:
        st.subheader("Trend Decomposition")
        beverage_to_decompose = st.selectbox(
            "Select a beverage for trend decomposition:", options=df_selection.columns
        )
        if beverage_to_decompose:
            st.plotly_chart(
                plotly_trend_decomposition(
                    df_selection[beverage_to_decompose], beverage_to_decompose
                ),
                use_container_width=True,
            )

# --- Forecasting Tab ---
with tabs[1]:
    st.header("Consumption Forecast")
    forecast_years = st.slider("Select number of years to forecast:", 1, 10, 5, 1)
    with st.spinner("Training model and generating forecast..."):
        fitted_model = cached_train_model(df_selection)
        forecast_df = generate_forecast(fitted_model, steps=forecast_years)
        st.plotly_chart(
            plotly_forecast(df_selection, forecast_df), use_container_width=True
        )
    st.subheader("Forecasted Values (Liters per capita)")
    st.dataframe(forecast_df)

# --- Backtesting Tab ---
with tabs[2]:
    st.header("Model Backtesting and Validation")
    st.markdown(
        """
        To validate the model's performance, we train it on data only up to **2019** and use it to forecast the period from **2020 to 2023**. 
        We then compare this forecast to the actual consumption data from that period.
        """
    )

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            # Split data
            train_df = df_selection[df_selection.index.year <= 2019]
            test_df = df_selection[df_selection.index.year > 2019]

            # Train model on historical data only
            backtest_model = train_var_model(train_df)

            # Forecast the test period
            backtest_forecast_df = generate_forecast(
                backtest_model, steps=len(test_df)
            )

            # Display plot
            st.plotly_chart(
                plotly_backtesting(test_df, backtest_forecast_df),
                use_container_width=True,
            )

            # Display dataframes
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Forecasted Values")
                st.dataframe(backtest_forecast_df)
            with col2:
                st.subheader("Actual Values")
                st.dataframe(test_df)

# --- What-If Analysis Tab ---
with tabs[3]:
    st.header("What-If Scenario Analysis")
    st.markdown(
        """
        This tool lets you see how a hypothetical shock in the last recorded year's consumption would affect future forecasts. 
        Adjust the sliders to simulate a change in consumption for 2023 and compare the resulting forecast to the original.
        """
    )

    # Original forecast for comparison
    original_model = cached_train_model(df_selection)
    original_forecast = generate_forecast(original_model, steps=5)

    # Create sliders for what-if scenario
    st.subheader("Simulate a Shock in 2023 Consumption")
    df_hypothetical = df_selection.copy()

    cols = st.columns(len(df_selection.columns))
    for i, col_name in enumerate(df_selection.columns):
        last_val = df_selection[col_name].iloc[-1]
        shock_val = cols[i].slider(
            f"Adjust {col_name.title()} consumption for 2023",
            min_value=float(last_val * 0.5),
            max_value=float(last_val * 1.5),
            value=float(last_val),
            step=0.1,
        )
        # Apply the shock to the last row
        df_hypothetical.iloc[-1, df_hypothetical.columns.get_loc(col_name)] = shock_val

    with st.spinner("Running what-if scenario..."):
        # Train a new model on the hypothetical data
        what_if_model = train_var_model(df_hypothetical)
        what_if_forecast = generate_forecast(what_if_model, steps=5)

        # Plot comparison
        st.plotly_chart(
            plotly_what_if_forecast(original_forecast, what_if_forecast),
            use_container_width=True,
        )

    # Display dataframes
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Forecast")
        st.dataframe(original_forecast)
    with col2:
        st.subheader("What-If Forecast")
        st.dataframe(what_if_forecast)