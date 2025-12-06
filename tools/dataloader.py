import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
from pathlib import Path
from typing import Optional


class KeywordSchema(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)
    required_kw: Optional[str]
    headers: Optional[str]


class KeywordSchemaRaw(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)


class ChatSchema(pa.DataFrameModel):
    Source: str
    Date1: str = pa.Field(nullable=True)
    Date2: str
    Time: str
    userPhone: str = pa.Field(coerce=True)
    quotedMessage: str = pa.Field(nullable=True)
    messageBody: str = pa.Field(nullable=True)
    mediaType: str = pa.Field(nullable=True)
    mediaCaption: str = pa.Field(nullable=True)
    Reason: str = pa.Field(nullable=True)


class DataLoader:

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.chats: list[DataFrame[ChatSchema]] = []

    def get_keyword_df(self, keyword_file: str) -> DataFrame[KeywordSchema]:
        keyword_path = self.base_path / keyword_file
        try:
            keywords = pd.read_excel(keyword_path).fillna("")
        except Exception as e:
            print("Error:", e)

        keywords["headers"] = keywords["brand"].str.cat(keywords["product"], "_")
        keywords["required_kw"] = ""
        df = KeywordSchema.validate(keywords)

        for idx in df.index:
            req_prod = str(df.at[idx, "required_product"])
            df.at[idx, "required_kw"] = DataLoader._get_required_kw(req_prod, df)
        return df

    @staticmethod
    def _get_required_kw(req_prod: str, df: DataFrame[KeywordSchema]) -> str:
        products = [p.strip() for p in req_prod.split(",")]
        keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
        return "|".join(keywords)

    def get_chat_dfs(self, chat_folder: str) -> dict[str, DataFrame[ChatSchema]]:
        sheets: dict[str, DataFrame[ChatSchema]] = {}
        chat_path = Path(self.base_path) / chat_folder
        subfolders = [f for f in chat_path.iterdir() if f.is_dir()]
        for sub in subfolders:
            print("Start processing:", sub.name)
            dataframes = self._get_chat_df_folder(sub)
            if dataframes:
                df = self._combine_chat(dataframes)
                sheets[sub.name] = df
        return sheets

    @classmethod
    def _get_chat_df_folder(cls, chat_path: Path) -> list[DataFrame[ChatSchema]]:
        dataframes: list[DataFrame[ChatSchema]] = []
        files = chat_path.iterdir()
        for file in files:
            df = cls._create_chat_df(file)
            if df is not None:
                dataframes.append(df)
        return dataframes

    @staticmethod
    def _create_chat_df(file: Path) -> DataFrame[ChatSchema] | None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print("File path error:", e)
            return None

        df["Source"] = file.name
        df["Reason"] = ""
        df = ChatSchema.validate(df)
        return df

    @staticmethod
    def _combine_chat(
        dataframes: list[DataFrame[ChatSchema]],
    ) -> DataFrame[ChatSchema]:
        combined = pd.concat(dataframes, ignore_index=True).reindex(
            [
                "Source",
                "Date1",
                "Date2",
                "Time",
                "userPhone",
                "quotedMessage",
                "messageBody",
                "mediaType",
                "mediaCaption",
                "Reason",
            ],
            axis=1,
        )
        validated_df = ChatSchema.validate(combined)
        return validated_df

    def output_to_xlsx(
        self, sheets: dict[str, DataFrame[ChatSchema]], filename: str
    ) -> None:
        output_path = Path(self.base_path) / filename
        with pd.ExcelWriter(output_path) as writer:
            for sheetname, dataframe in sheets.items():
                dataframe.to_excel(writer, sheet_name=sheetname, index=False)
                print(f"Sheet: {sheetname} has been added to file.")
        print("Output of file has been done.")
