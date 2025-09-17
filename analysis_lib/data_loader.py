import pandas as pd

def load_and_prepare_data(filepath="Consumption of alcoholic beverages in Russia 1998-2023.csv"):
    """
    Loads the alcohol consumption data from a CSV file and prepares it for analysis.
    """
    df = pd.read_csv(filepath)
    
    # Use the correct capitalized column name 'Year'
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index('Year', inplace=True)

    # Pivot the table to get beverage types as columns
    df_pivot = df.pivot_table(
        index=df.index, 
        columns='Type', 
        values='Consumption of alcoholic beverages (in liters per capita)'
    )

    # --- FIX: Implement a more robust column cleaning method ---
    # Create a mapping for desired column names
    column_rename_map = {
        'Wine': 'wine',
        'Beer and Сider': 'beer',
        'Vodka and Liqueurs': 'vodka',
        'Brandy': 'brandy'
    }
    df_pivot.rename(columns=column_rename_map, inplace=True)
    
    # Select only the columns we need for the analysis
    final_df = df_pivot[['wine', 'beer', 'vodka', 'brandy']]

    # Ensure all data is numeric and handle any potential errors
    for col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
    final_df.dropna(inplace=True)
    
    return final_df.copy()