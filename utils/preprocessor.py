from pathlib import Path
import pandera as pa
from pandera.typing import DataFrame
from utils.validator import KeywordSchemaRaw, KeywordSchema, ChatSchemaRaw, ChatSchema
from utils.loader import DataLoader
import pandas as pd


class Preprocessor:

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)

    def get_keyword_df(self, file_path: str | Path) -> DataFrame[KeywordSchema] | None:
        keyword_path = self.base_path / file_path
        df = DataLoader.xlsx_to_df(keyword_path)
        if df is not None:
            try:
                keywords = KeywordSchemaRaw.validate(df.fillna(""))
            except Exception as e:
                print("Error:", e)
                return None
        
        if keywords is not None:
            keywords["headers"] = keywords["brand"].str.cat(keywords["product"], "_")
            keywords["required_keyword"] = ""
            validated_keywords = KeywordSchema.validate(keywords)

            for idx in validated_keywords.index:
                req_prod = str(validated_keywords.at[idx, "required_product"])
                validated_keywords.at[idx, "required_keyword"] = Preprocessor._get_required_keyword(req_prod, validated_keywords)
            return validated_keywords
        
        return None

    @staticmethod
    def _get_required_keyword(req_prod: str, df: DataFrame[KeywordSchema]) -> str:
        products = [p.strip() for p in req_prod.split(",")]
        keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
        return "|".join(keywords)

    def get_chat_df_dict(self, chat_folder: str) -> dict[str, DataFrame[ChatSchema]]:
        sheets: dict[str, DataFrame[ChatSchema]] = {}
        chat_path = self.base_path / chat_folder
        subfolders = [f for f in chat_path.iterdir() if f.is_dir()]
        for sub in subfolders:
            print("Start processing:", sub.name)
            dataframes = Preprocessor._get_chat_df_folder(sub)
            if dataframes:
                df = Preprocessor._combine_chat(dataframes).reindex(
            [
                "Source",
                "Group",
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
                sheets[sub.name] = df
        return sheets

    @staticmethod
    def _get_chat_df_folder(chat_path: Path) -> list[DataFrame[ChatSchema]]:
        dataframes: list[DataFrame[ChatSchema]] = []
        files = chat_path.iterdir()
        for file in files:
            df = Preprocessor.get_chat_df(file)
            if df is not None:
                dataframes.append(df)
        return dataframes

    @staticmethod
    def get_chat_df(file_path: Path) -> DataFrame[ChatSchema] | None:
        df = DataLoader.csv_to_df(file_path)
        if df is not None:
            chat = ChatSchemaRaw.validate(df.fillna(""))

        if chat is not None:
            chat["Source"] = file_path.name
            chat["Group"] = ""
            chat["Reason"] = ""
            validated_chat = ChatSchema.validate(chat)
            return validated_chat

        return None

    @staticmethod
    def _combine_chat(
        dataframes: list[DataFrame[ChatSchema]],
    ) -> DataFrame[ChatSchema]:
        combined = pd.concat(dataframes, ignore_index=True)
        validated_df = ChatSchema.validate(combined)
        return validated_df
