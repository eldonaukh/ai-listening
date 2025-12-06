import pandas as pd
import json
from pathlib import Path
from tools.ai import get_analyzer, SentimentAnalyzer
from tools.dataloader import KeywordSchema, ChatSchema, DataLoader
from typing import Any, NamedTuple, cast
from tqdm import tqdm
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

    
class KeywordRow(NamedTuple):
    brand: str
    product: str
    keyword: str
    required_product: str | None
    required_kw: str | None


def pass_to_llm(row: pd.Series, header: str, keywords: str) -> pd.Series:
    data = (
        f"Formula Brand: {header}, Keyword: {keywords}, Message: {row["messageBody"]}"
    )
    analyzer = get_analyzer("poe", "GPT-5-nano")
    response = analyzer.analyze(data)
    row[header] = response["sentiment"]
    row["Reason"] = response["reason"]
    return row


def process_chat(df: DataFrame[ChatSchema], keyword_df: DataFrame[KeywordSchema]) -> DataFrame[ChatSchema]:
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
                mask_required = df["messageBody"].str.contains(
                    row.required_kw, case=False, na=False
                )
                mask_keyword = df["messageBody"].str.contains(
                    row.keyword, case=False, na=False
                )
                mask = mask_required & mask_keyword
            else:
                mask = df["messageBody"].str.contains(row.keyword, case=False, na=False)
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
                mask_required = df.loc[~mask_skip, "messageBody"].str.contains(
                    row.required_kw, case=False, na=False
                )
                mask_keyword = df.loc[~mask_skip, "messageBody"].str.contains(
                    row.keyword, case=False, na=False
                )
                mask = mask_required & mask_keyword
            else:
                mask = df.loc[~mask_skip, "messageBody"].str.contains(
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

class ChatProcessor:
    
    def __init__(self, keyword_df: DataFrame[KeywordSchema], chat_df: DataFrame[ChatSchema]):
        self._keyword_df = keyword_df
        self._chat_df = chat_df
    
    @property
    def unique_headers(self) -> list[str]:
        return list(self._keyword_df["headers"].unique())
    
    @property
    def generic_headers(self) -> list[str]:
        return [header for header in self.unique_headers if "generic" in header]
    
    @property
    def non_generic_headers(self) -> list[str]:
        return [header for header in self.unique_headers if "generic" not in header]
    
    def get_keywords_of_header(self, header: str) -> DataFrame[KeywordSchema]:
        return self._keyword_df[self._keyword_df["headers"] == header]
    


def main() -> None:
    # get keywords from xlsx    
    loader = DataLoader("./files")
    keyword_df = loader.get_keyword_df("keywords.xlsx")
    
    # load all csv files in folders into one df
    sheets = loader.get_chat_dfs("./chats")
    # chat = base_path / "chats"
    # subfolders: list[Path] = [f for f in chat.iterdir() if f.is_dir()]
    # for sub in subfolders:
    #     print("Start processing:", sub.name)
    #     loader = DataLoader(base_path, "keywords.xlsx")
    #     loader.get_chat_df_folder(sub)        
    #     df: DataFrame[ChatSchema] | None = loader.combine_chat()
    if sheets is not None:
        for sheet, df in sheets.items():
            processed = process_chat(df=df, keyword_df=keyword_df)
            sheets[sheet] = processed
            print("Processed:", sheet)

    # export df into xlsx
    loader.output_to_xlsx(sheets, filename="output.xlsx")



if __name__ == "__main__":
    main()
