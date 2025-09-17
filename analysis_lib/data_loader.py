import pandas as pd

def load_and_prepare_data(filepath="Consumption of alcoholic beverages in Russia 1998-2023.csv"):
    """
    Loads the alcohol consumption data from a CSV file and prepares it for analysis.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A cleaned and prepared DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(filepath)

    # --- FIX: Use the correct capitalized column name 'Year' ---
    # Convert 'Year' to datetime objects and set as index
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index('Year', inplace=True)

    # Pivot the table to get beverage types as columns
    # We will use the main consumption column for the VAR model
    df_pivot = df.pivot_table(
        index=df.index, 
        columns='Type', 
        values='Consumption of alcoholic beverages (in liters per capita)'
    )

    # Clean up column names for easier access (e.g., "Vodka and Liqueurs" -> "vodka")
    df_pivot.columns = [col.split()[0].lower() for col in df_pivot.columns]
    
    # Select the main categories used in the original notebook
    final_df = df_pivot[['wine', 'beer', 'vodka', 'brandy']]

    # Ensure all data is numeric, converting non-numeric to NaN
    for col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    # Drop rows with any missing values
    final_df.dropna(inplace=True)

    return final_df