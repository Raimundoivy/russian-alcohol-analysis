import pandas as pd
from pandas import DataFrame


def load_and_prepare_data(
    filepath: str = "Consumption of alcoholic beverages in Russia 1998-2023.csv",
) -> DataFrame:
    """
    Loads the alcohol consumption data from a CSV file and prepares it for analysis.

    Args:
        filepath (str, optional): The path to the CSV file. Defaults to "Consumption of alcoholic beverages in Russia 1998-2023.csv".

    Returns:
        DataFrame: A pandas DataFrame with the prepared data.
    """
    df = pd.read_csv(filepath)

    df["Year"] = pd.to_datetime(df["Year"], format="%Y")
    df.set_index("Year", inplace=True)

    df_pivot = df.pivot_table(
        index=df.index,
        columns="Type",
        values="Consumption of alcoholic beverages (in liters per capita)",
    )

    column_rename_map = {
        "Wine": "wine",
        "Beer and Сider": "beer",
        "Vodka and Liqueurs": "vodka",
        "Brandy": "brandy",
    }
    df_pivot.rename(columns=column_rename_map, inplace=True)

    # Explicitly create a copy to avoid SettingWithCopyWarning
    final_df = df_pivot[["wine", "beer", "vodka", "brandy"]].copy()

    # Use .loc to safely modify the DataFrame
    for col in final_df.columns:
        final_df.loc[:, col] = pd.to_numeric(final_df[col], errors="coerce")

    final_df.dropna(inplace=True)

    return final_df