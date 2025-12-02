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


class ChatSchema(pa.DataFrameModel):
    Source: str
    Date1: str
    Date2: str
    Time: str
    UserPhone: str
    UserName: str
    QuotedMessage: str = pa.Field(nullable=True)
    MessageBody: str = pa.Field(nullable=True)
    MediaType: str = pa.Field(nullable=True)
    MediaCaption: str = pa.Field(nullable=True)
    Reason: str = pa.Field(nullable=True)


class DataLoader:

    def __init__(self, base_path: Path, keyword_file: str) -> None:
        self.base_path = base_path
        self.keyword_file = keyword_file
        self.chat_path = self.base_path / "chats"
        self.keyword_path = self.base_path / keyword_file
        self.chats: list[DataFrame[ChatSchema]] = []

    def get_keyword_df(self) -> DataFrame[KeywordSchema]:

        try:
            keywords = pd.read_excel(self.keyword_path).fillna("")
        except Exception as e:
            print("Error:", e)

        keywords["headers"] = keywords["brand"].str.cat(keywords["product"], "_")
        keywords["required_kw"] = ""
        df = KeywordSchema.validate(keywords)

        for idx in df.index:
            req_prod = str(df.at[idx, "required_product"])
            df.at[idx, "required_kw"] = self._get_required_kw(req_prod, df)
        return df

    def _get_required_kw(self, req_prod: str, df: DataFrame[KeywordSchema]):
        products = [p.strip() for p in req_prod.split(",")]
        keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
        return "|".join(keywords)

    def _get_chat_df(self, file: Path) -> DataFrame[ChatSchema] | None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print("File path error:", e)
            return None

        df["Source"] = file.name
        df = ChatSchema.validate(df)
        return df

    def get_chat_df_all(self) -> None:
        files = self.chat_path.iterdir()
        for file in files:
            df = self._get_chat_df(file)
            if df is not None:
                self.chats.append(df)

    def combine_chat(self) -> DataFrame[ChatSchema] | None:
        if len(self.chats) == 0:
            return None
        combined = pd.concat(self.chats, ignore_index=True).reindex(
            [
                "Source",
                "Date1",
                "Date2",
                "Time",
                "UserPhone",
                "UserName",
                "QuotedMessage",
                "MessageBody",
                "MediaType",
                "MediaCaption",
                "Reason",
            ],
            axis=1,
        )
        validated_df = ChatSchema.validate(combined)
        return validated_df
