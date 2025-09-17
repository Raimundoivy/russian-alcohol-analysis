import unittest
import pandas as pd
from analysis_lib.data_loader import load_and_prepare_data


class TestDataLoader(unittest.TestCase):
    def test_load_and_prepare_data(self):
        """
        Test the load_and_prepare_data function.
        """
        df = load_and_prepare_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df.columns), 4)
        self.assertIn("wine", df.columns)
        self.assertIn("beer", df.columns)
        self.assertIn("vodka", df.columns)
        self.assertIn("brandy", df.columns)


if __name__ == "__main__":
    unittest.main()