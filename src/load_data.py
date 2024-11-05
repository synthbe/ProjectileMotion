import pandas as pd
import glob
import os

TABLES_DIR: str = r'./data'

def read_tables(tables_dir=TABLES_DIR) -> pd.DataFrame:
    """
    Reads data directory with all the tables and concate them
    """
    tables_file: list = glob.glob(os.path.join(tables_dir, "*.csv"))
    dfs: list = [pd.read_csv(file, sep=', ', engine='python') for file in tables_file]
    df_combined = pd.concat(dfs, ignore_index=True)

    return df_combined