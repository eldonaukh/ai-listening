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
    
    @staticmethod
    def df_dict_to_xlsx(
        dataframes: dict[str, pd.DataFrame], output_path: str | Path
    ):
        with pd.ExcelWriter(output_path) as writer:
            for sheetname, dataframe in dataframes.items():
                dataframe.to_excel(writer, sheet_name=sheetname, index=False)
                print(f"Sheet: {sheetname} has been added to file.")
        print("Output of file has been done.")
