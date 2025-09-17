import streamlit as st
import matplotlib.pyplot as plt
from analysis_lib.data_loader import load_and_prepare_data
from analysis_lib.forecasting_models import train_var_model, generate_forecast, get_granger_causality_results
from analysis_lib.plotting import plot_correlation_heatmap, plot_forecast, plot_residuals, plot_trend_decomposition

# --- Page Configuration ---
st.set_page_config(page_title="Russian Alcohol Consumption Analysis", layout="wide")

# --- Caching Functions ---
@st.cache_data
def cached_load_data():
    return load_and_prepare_data()

@st.cache_resource
def cached_train_model(_df):
    return train_var_model(_df)

# --- Main Application ---
st.title("Interactive Analysis of Alcohol Consumption in Russia")

# --- Load Data ---
try:
    df = cached_load_data()
except FileNotFoundError:
    st.error("Error: The dataset file was not found.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
selected_beverages = st.sidebar.multiselect(
    "Select beverages to analyze:",
    options=df.columns,
    default=list(df.columns)
)

if not selected_beverages:
    st.warning("Please select at least one beverage to continue.")
    st.stop()

df_selection = df[selected_beverages]

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Exploratory Data Analysis", "🍻 Forecasting", "🔬 Model Diagnostics", "🍾 Beverage Deep Dive"])

with tab1:
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Per Capita Consumption (Liters)")
        st.dataframe(df_selection)
    with col2:
        st.markdown("#### Correlation Matrix")
        if len(df_selection.columns) > 1:
            st.pyplot(plot_correlation_heatmap(df_selection))
        else:
            st.info("Select at least two beverages to see their correlation.")

with tab2:
    st.header("Consumption Forecast")
    forecast_years = st.slider("Select number of years to forecast:", 1, 10, 5, 1)
    with st.spinner('Training model and generating forecast...'):
        fitted_model = cached_train_model(df_selection)
        forecast_df = generate_forecast(fitted_model, steps=forecast_years)
        st.pyplot(plot_forecast(df_selection, forecast_df))
    st.markdown("#### Forecasted Values (Liters per capita)")
    st.dataframe(forecast_df)

with tab3:
    st.header("Model Diagnostics")
    st.markdown("This section provides tools to evaluate the performance and assumptions of the VAR model.")
    
    if len(df_selection.columns) < 2:
        st.warning("Please select at least two beverages to view model diagnostics.")
    else:
        fitted_model_diag = cached_train_model(df_selection)
        
        st.subheader("Residual Plots")
        st.markdown("Residuals are the errors of the model. Ideally, they should be random noise with no discernible pattern, centered around zero.")
        st.pyplot(plot_residuals(fitted_model_diag))
        
        st.subheader("Granger Causality Test")
        st.markdown("This test checks if the past values of one beverage can help predict the future values of another. A low p-value (e.g., < 0.05) suggests that one beverage 'Granger-causes' another.")
        causality_results = get_granger_causality_results(df_selection, selected_beverages)
        for (var1, var2), p_value in causality_results.items():
            result_text = f"**{var1.title()}** does {'**not** ' if p_value > 0.05 else ''}Granger-cause **{var2.title()}**"
            st.metric(label=result_text, value=f"{p_value:.4f}", help="P-value for the F-test. Lower is more significant.")

with tab4:
    st.header("Beverage Deep Dive")
    st.markdown("Analyze the long-term trend of a single beverage.")
    
    beverage_to_decompose = st.selectbox(
        "Select a beverage for trend decomposition:",
        options=df_selection.columns
    )
    
    if beverage_to_decompose:
        st.pyplot(plot_trend_decomposition(df_selection[beverage_to_decompose], beverage_to_decompose))