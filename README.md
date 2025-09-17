# **Russian Alcohol Consumption Analysis (1998-2023)**

## **Introduction: Business Problem**

This project provides an in-depth analysis of alcohol consumption trends in Russia from 1998 to 2023. The primary goal of this analysis is to understand the dynamics of alcohol consumption in Russia and to build a forecasting model to predict future trends. This can provide insights into public health, economic factors, and cultural shifts over time. From a business perspective, this analysis could be valuable for a beverage company looking to optimize its product portfolio and forecast demand in the Russian market. Similarly, a public health organization could use these findings to monitor consumption trends and inform policy decisions.

## **Data Source**

The dataset used in this analysis is publicly available on Kaggle. It contains yearly per capita consumption data for various alcoholic beverages in Russia from 1998 to 2023.

* **Source**: [Russian Alcohol Consumption and Deaths](https://www.kaggle.com/datasets/raimundoivy/russian-alcohol-consumption-and-deaths)
* **File**: `Consumption of alcoholic beverages in Russia 1998-2023.csv`

## **Key Visualizations**

### **Consumption Forecast**

![Consumption Forecast](consumption_forecast.png)

### **Correlation Heatmap**

![Correlation Heatmap](correlation_heatmap.png)

### **Backtesting and Validation**

![Backtesting Validation](backtesting_validation_plot.png)

## **Methodology**

The analysis is conducted in the `main.py` script and follows these key steps:

1.  **Data Preprocessing**: The dataset is loaded, cleaned, and transformed from a long to a wide format.
2.  **Stationarity Testing**: The Augmented Dickey-Fuller (ADF) test is used to check for stationarity.
3.  **Vector Autoregression (VAR) Model**:
    * **Model Selection**: The optimal lag order is determined using AIC and BIC.
    * **Forecasting**: A VAR model is fitted to forecast future consumption.
    * **Impulse Response Function (IRF)**: To analyze how a shock in one variable affects others.
4.  **VARMAX Model**: The analysis is extended with a VARMAX model to include the COVID-19 pandemic as an exogenous variable.

## **Getting Started**

### **Prerequisites**

* Python 3.x
* pip

### **Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Raimundoivy/russian-alcohol-analysis](https://github.com/Raimundoivy/russian-alcohol-analysis)
    cd russian-alcohol-analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **Running the Analysis**

To run the complete analysis pipeline, execute the `main.py` script:

```bash
python main.py