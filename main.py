import pandas as pd
from utils.ai import get_analyzer
from utils.validator import KeywordSchema, ChatSchema, KeywordRow
from utils.preprocessor import Preprocessor
from utils.chatprocessor import ChatProcessor
from utils.loader import DataLoader
from typing import Any, cast
from pandera.typing import DataFrame


# def pass_to_llm(row: pd.Series, header: str, keywords: str) -> pd.Series:
#     data = (
#         f"Formula Brand: {header}, Keyword: {keywords}, Message: {row["messageBody"]}"
#     )
#     analyzer = get_analyzer("poe", "GPT-5-nano")
#     response = analyzer.analyze(data)
#     row[header] = response.sentiment
#     row["Reason"] = response.reason
#     return row


# def process_chat(
#     df: DataFrame[ChatSchema], keyword_df: DataFrame[KeywordSchema]
# ) -> DataFrame[ChatSchema]:
#     # add keyword cols to df
#     headers: list[str] = list(keyword_df["headers"].unique())
#     for header in headers:
#         df[header] = 0

#     # get lists of col headers, depending on generic or specific product
#     generic = [header for header in headers if "generic" in header]
#     non_generic = [header for header in headers if "generic" not in header]

#     # check keywords of non-generic product column
#     for header in non_generic:
#         matched = keyword_df[keyword_df["headers"] == header]
#         for row in cast(list[KeywordRow], matched.itertuples(index=False)):
#             if row.required_keyword:
#                 mask_required = df["messageBody"].str.contains(
#                     row.required_keyword, case=False, na=False
#                 )
#                 mask_keyword = df["messageBody"].str.contains(
#                     row.keyword, case=False, na=False
#                 )
#                 mask = mask_required & mask_keyword
#             else:
#                 mask = df["messageBody"].str.contains(row.keyword, case=False, na=False)
#             df[header] = df[header] | mask.astype(int)

#     # check keywords of generic product column
#     for header in generic:
#         main = header.replace("_generic", "")
#         subbrand = [brand for brand in non_generic if main in brand]
#         # get rows where non-generic product columns are tagged
#         mask_skip = df[subbrand].any(axis=1)
#         matched = keyword_df[keyword_df["headers"] == header]
#         for row in cast(list[KeywordRow], matched.itertuples(index=False)):
#             if row.required_keyword:
#                 mask_required = df.loc[~mask_skip, "messageBody"].str.contains(
#                     row.required_keyword, case=False, na=False
#                 )
#                 mask_keyword = df.loc[~mask_skip, "messageBody"].str.contains(
#                     row.keyword, case=False, na=False
#                 )
#                 mask = mask_required & mask_keyword
#             else:
#                 mask = df.loc[~mask_skip, "messageBody"].str.contains(
#                     row.keyword, case=False, na=False
#                 )
#             df.loc[~mask_skip, header] = df.loc[~mask_skip, header] | mask.astype(int)

#     for header in headers:
#         # turn 0 into empty str
#         df[header] = df[header].astype("object")
#         df.loc[df[header] == 0, header] = ""

#     for header in headers:
#         # get header keyword list in str
#         keywords: list[str] = []
#         matched = keyword_df[keyword_df["headers"] == header]
#         for kw_row in matched.itertuples():
#             row_req_kw: Any = kw_row.required_keyword
#             if row_req_kw:
#                 keywords.extend(row_req_kw.split("|"))
#             row_kw: Any = kw_row.keyword
#             keywords.append(row_kw)
#         kw_str = ", ".join(set(keywords))

#         # ask llm for sentiment check
#         df[df[header] == 1] = df[df[header] == 1].apply(
#             lambda row: pass_to_llm(row, header, kw_str), axis=1
#         )

#     return df


def main() -> None:
    p = Preprocessor("./data")
    chats = p.get_chat_df_dict("chats")
    keyword = p.get_keyword_df("keywords.xlsx")
    # print(chats)
    # print(keyword)
    a = get_analyzer("poe", "gemini-2.5-flash")
    if keyword is not None:
        c = ChatProcessor(keyword_df=keyword, analyzer=a)
        # print(c.keyword_df.to_dict(orient="records"))

        for sheet, chat in chats.items():
            df = c.process_chat_df(chat)
            chats[sheet] = df

        c.save_result(chats, "./data/output.xlsx")


if __name__ == "__main__":
    main()
