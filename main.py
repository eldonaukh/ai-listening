import pandas as pd
import json
from pathlib import Path
from tools.ai import sentiment_check
from typing import Any, NamedTuple, cast
from tqdm import tqdm
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

class KeywordSchema(pa.DataFrameModel):
    id: Series[int]
    brand: Series[str]
    product: Series[str]
    keyword: Series[str]
    required_product: Series[str] = pa.Field(nullable=True)
    required_kw: Series[str] = pa.Field(nullable=True)

class ChatSchema(pa.DataFrameModel):
    id: Series[int]
    Source: Series[str]
    Date1: Series[str]
    Date2: Series[str]
    Time: Series[str]
    UserPhone: Series[str]
    UserName: Series[str]
    QuotedMessage: Series[str] = pa.Field(nullable=True)
    MessageBody: Series[str] = pa.Field(nullable=True)
    MediaType: Series[str] = pa.Field(nullable=True)
    MediaCaption: Series[str] = pa.Field(nullable=True)
    Reason: Series[str] = pa.Field(nullable=True)
    
class KeywordRow(NamedTuple):
    brand: str
    product: str
    keyword: str
    required_product: str | None
    required_kw: str | None

def process_df_keywords(df: DataFrame[KeywordSchema]) -> DataFrame[KeywordSchema]:
    return df

def process_df_chat(df: DataFrame[ChatSchema]) -> DataFrame[ChatSchema]:
    return df

def load_csv_into_df(path: Path) -> pd.DataFrame | None:
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
    validated_combined = process_df_chat(validated_df)
    print(f"Processed {len(processed)} files in folder.")
    return validated_combined


# combine required product keyword
def get_req_product_kw(row: pd.Series, df: pd.DataFrame) -> str | None:
    if pd.isna(row["required_product"]):
        return None
    products = [p.strip() for p in row["required_product"].split(",")]
    keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
    return "|".join(keywords)


def pass_to_llm(row: pd.Series, header: str, keywords: str) -> pd.Series:
    data = (
        f"Formula Brand: {header}, Keyword: {keywords}, Message: {row["MessageBody"]}"
    )
    response = sentiment_check(data)
    response_json = json.loads(response)
    row[header] = response_json["sentiment"]
    row["Reason"] = response_json["reason"]
    return row


def process_chat(df: DataFrame[ChatSchema], keyword_df: DataFrame[KeywordSchema]) -> pd.DataFrame:
    # add keyword cols to df
    headers: list[str] = list(keyword_df["headers"].unique())
    for header in headers:
        df[header] = 0

    # get lists of col headers, depending on generic or specific product
    generic = [header for header in headers if "generic" in header]
    non_generic = [header for header in headers if "generic" not in header]

    # check keywords of non-generic product column
    for header in non_generic:
        matched = keyword_df[keyword_df["headers"] == header]
        for row in cast(list[KeywordRow], matched.itertuples(index=False)):          
            if row.required_kw:
                mask_required = df["MessageBody"].str.contains(
                    row.required_kw, case=False, na=False
                )
                mask_keyword = df["MessageBody"].str.contains(
                    row.keyword, case=False, na=False
                )
                mask = mask_required & mask_keyword
            else:
                mask = df["MessageBody"].str.contains(row.keyword, case=False, na=False)
            df[header] = df[header] | mask.astype(int)

    # check keywords of generic product column
    for header in generic:
        main = header.replace("_generic", "")
        subbrand = [brand for brand in non_generic if main in brand]
        # get rows where non-generic product columns are tagged
        mask_skip = df[subbrand].any(axis=1)
        matched = keyword_df[keyword_df["headers"] == header]
        for row in cast(list[KeywordRow], matched.itertuples(index=False)):
            if row.required_kw:
                mask_required = df.loc[~mask_skip, "MessageBody"].str.contains(
                    row.required_kw, case=False, na=False
                )
                mask_keyword = df.loc[~mask_skip, "MessageBody"].str.contains(
                    row.keyword, case=False, na=False
                )
                mask = mask_required & mask_keyword
            else:
                mask = df.loc[~mask_skip, "MessageBody"].str.contains(
                    row.keyword, case=False, na=False
                )
            df.loc[~mask_skip, header] = df.loc[~mask_skip, header] | mask.astype(int)

    for header in headers:
        # turn 0 into empty str
        df[header] = df[header].astype("object")
        df.loc[df[header] == 0, header] = ""

    for header in headers:
        # get header keyword list in str
        keywords: list[str] = []
        matched = keyword_df[keyword_df["headers"] == header]
        for kw_row in matched.itertuples():
            row_req_kw: Any = kw_row.required_kw
            if row_req_kw:
                keywords.extend(row_req_kw.split("|"))
            row_kw: Any = kw_row.keyword
            keywords.extend(row_kw)
        kw_str = ", ".join(set(keywords))

        # ask llm for sentiment check
        df[df[header] == 1] = df[df[header] == 1].apply(
            lambda row: pass_to_llm(row, header, kw_str), axis=1
        )

    return df


def main() -> None:
    # get keywords from xlsx
    base_path = Path("./files")
    keyword_df = pd.read_excel(base_path / "keywords.xlsx")
    keyword_df["headers"] = keyword_df["brand"] + "_" + keyword_df["product"]
    keyword_df["required_kw"] = keyword_df.apply(
        lambda row: get_req_product_kw(row, keyword_df), axis=1
    )
    sheets: dict[str, pd.DataFrame] = {}
    # load all csv files in folder into one df
    chat = base_path / "chat"
    subfolders: list[Path] = [f for f in chat.iterdir() if f.is_dir()]
    for sub in subfolders:
        print("Start processing:", sub.name)
        df: pd.DataFrame | None = load_csv_into_df(sub)
        if df:
            processed = process_chat(df=df, keyword_df=keyword_df)
            print("Processed:", sub.name)
            sheets[sub.name] = processed

    # export df into xlsx
    with pd.ExcelWriter("./files/output.xlsx") as writer:
        for sheetname, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=sheetname, index=False)
            print(f"Sheet: {sheetname} has been added to file.")


if __name__ == "__main__":
    main()
