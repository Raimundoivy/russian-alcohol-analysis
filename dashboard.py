import streamlit as st
import matplotlib.pyplot as plt
from analysis_lib.data_loader import load_and_prepare_data
from analysis_lib.forecasting_models import train_var_model, generate_forecast
from analysis_lib.plotting import plot_correlation_heatmap, plot_forecast

# --- Page Configuration ---
st.set_page_config(
    page_title="Russian Alcohol Consumption Analysis",
    layout="wide"
)

# --- Main Application ---

st.title("Analysis of Alcohol Consumption in Russia (1998-2023)")

st.markdown("""
This dashboard presents an analysis of alcohol consumption trends in Russia, using a Vector Autoregression (VAR) model to forecast future consumption. 
All data loading, modeling, and plotting are performed by a modular Python library, and the results are displayed here.
""")

# --- Load Data ---
# Use a spinner to show that data is being loaded.
with st.spinner('Loading and preparing data...'):
    try:
        df = load_and_prepare_data()
        st.success("Data loaded successfully!")
    except FileNotFoundError:
        st.error("Error: The dataset file was not found. Please make sure 'Consumption of alcoholic beverages in Russia 1998-2023.csv' is in the project directory.")
        st.stop()

# --- Display Data and EDA ---
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Per Capita Consumption (Liters)")
    st.dataframe(df)

with col2:
    st.markdown("#### Correlation Matrix")
    # Generate and display the correlation heatmap
    fig_heatmap = plot_correlation_heatmap(df)
    st.pyplot(fig_heatmap)

# --- Forecasting ---
st.header("5-Year Consumption Forecast")

# Add a slider to make the forecast interactive
forecast_years = st.slider(
    "Select number of years to forecast:",
    min_value=1,
    max_value=10,
    value=5, # Default value
    step=1
)

with st.spinner('Training model and generating forecast...'):
    # Train the model
    fitted_model = train_var_model(df)
    
    # Generate the forecast
    forecast_df = generate_forecast(fitted_model, steps=forecast_years)
    
    # Plot the forecast
    fig_forecast = plot_forecast(df, forecast_df)
    st.pyplot(fig_forecast)

st.markdown("#### Forecasted Values (Liters per capita)")
st.dataframe(forecast_df)