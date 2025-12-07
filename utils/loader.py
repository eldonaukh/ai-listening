import pandas as pd
from pathlib import Path


class DataLoader:

    @staticmethod
    def xlsx_to_df(path: str | Path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            print("Error:", e)
            
    @staticmethod
    def csv_to_df(path: str | Path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print("Error:", e)