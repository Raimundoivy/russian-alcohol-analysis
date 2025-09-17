import unittest
import pandas as pd
from analysis_lib.data_loader import load_and_prepare_data
from analysis_lib.forecasting_models import train_var_model, generate_forecast
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper


class TestForecastingModels(unittest.TestCase):
    def setUp(self):
        """
        Set up the test data.
        """
        self.df = load_and_prepare_data()

    def test_train_var_model(self):
        """
        Test the train_var_model function.
        """
        fitted_model = train_var_model(self.df)
        self.assertIsInstance(fitted_model, VARResultsWrapper)

    def test_generate_forecast(self):
        """
        Test the generate_forecast function.
        """
        fitted_model = train_var_model(self.df)
        forecast_df = generate_forecast(fitted_model, steps=5)
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 5)
        self.assertEqual(len(forecast_df.columns), 4)


if __name__ == "__main__":
    unittest.main()