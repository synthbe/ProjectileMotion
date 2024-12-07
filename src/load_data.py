import pandas as pd
from pandas import DataFrame
from typing import TypeAlias, Tuple
from numpy import typing, float64
import glob
import os

FloatArray: TypeAlias = typing.NDArray[float64]

TABLES_DIR: str = r'./data'

def read_tables(tables_dir: str=TABLES_DIR) -> pd.DataFrame:
    """
    Reads data directory with all the tables, concate them and returns ad a pandas DataFrame
    """

    tables_file = glob.glob(os.path.join(tables_dir, "*.csv"))
    dfs = [pd.read_csv(file, sep=', ', engine='python') for file in tables_file]
    df_combined = pd.concat(dfs, ignore_index=True)

    return df_combined

def tidy_and_split_data(df: DataFrame) -> Tuple[DataFrame, FloatArray, FloatArray]:
    """
    Returns the dataframe as tidy pattern and splited
    """

    tidy_df = df.melt(id_vars=['h'], var_name='H', value_name='R')
    tidy_df = tidy_df.map(float).sort_values(by='H')
    tidy_df.sample(frac=1)
    tidy_df.dropna(inplace=True)

    X = tidy_df.drop(columns=['R'])
    y = tidy_df.R

    return tidy_df, X.values, y.values
