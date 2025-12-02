import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from pathlib import Path
from typing import NamedTuple, Any, cast, Optional


class KeywordSchema(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)
    required_kw: Optional[str]
    headers: Optional[str]


class ChatSchema(pa.DataFrameModel):
    id: int
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

def get_keyword_df(base_path: Path, filename: str) -> DataFrame[KeywordSchema]:

    try:
        keywords = pd.read_excel(base_path / filename).fillna("")
    except Exception as e:
        print("Error:", e)

    keywords["headers"] = keywords["brand"].str.cat(keywords["product"], "_")
    keywords["required_kw"] = ""
    df = KeywordSchema.validate(keywords)

    for idx in df.index:
        req_prod = str(df.at[idx, "required_product"])
        df.at[idx, "required_kw"] = get_required_kw(req_prod, df)
    return df


def get_required_kw(req_prod: str, df: DataFrame[KeywordSchema]):
    products = [p.strip() for p in req_prod.split(",")]
    keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
    return "|".join(keywords)


def get_chat_df(path: Path) -> pd.DataFrame | None:
    files = path.iterdir()
    df_list = []
    processed = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df["Source"] = file.name
            df_list.append(df)
            processed.append(file.name)
        except Exception as e:
            print("File path error:", e)
            return None

    combined = pd.concat(df_list, ignore_index=True).reindex(
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
    print(f"Processed {len(processed)} files in folder.")
    return validated_df
