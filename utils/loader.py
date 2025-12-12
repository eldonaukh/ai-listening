from pathlib import Path

import pandas as pd


class DataLoader:

    @staticmethod
    def xlsx_to_df(path: str | Path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise Exception("Error:", e)

    @staticmethod
    def csv_to_df(path: str | Path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise Exception("Error:", e)