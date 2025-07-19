import pandas as pd
from pathlib import Path
from tools.ai import sentiment_check
from typing import Any


def load_csv_into_df(path: str = "./files/chat/") -> pd.DataFrame | None:
    files = Path(path).iterdir()
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
    print(f"Processed {len(processed)} files")
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
        ],
        axis=1,
    )
    return combined


# combine required product keyword
def get_req_product_kw(row: pd.Series, df: pd.DataFrame) -> str | None:
    if pd.isna(row["required_product"]):
        return None
    products = [p.strip() for p in row["required_product"].split(",")]
    keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
    return "|".join(keywords)


def pass_to_llm(row: pd.Series, header: str, keyword_df: pd.DataFrame) -> pd.Series:
    keywords: list[str] = []
    matched = keyword_df[keyword_df["headers"] == header]
    for kw_row in matched.itertuples():
        row_req_kw: Any = kw_row.required_kw
        if row_req_kw:
            keywords.extend(row_req_kw.split("|"))
        row_kw: Any = kw_row.keyword
        keywords.extend(row_kw)

    data = f'{"Formula Brand": header, "Keyword": ", ".join(set(keywords)), "Message": row["MessageBody"]}'
    sentiment = sentiment_check(data)
    row["header"] = sentiment

    return row


def main():
    # get keywords from xlsx
    keyword_df = pd.read_excel("./files/keywords.xlsx")
    keyword_df["headers"] = keyword_df["brand"] + "_" + keyword_df["product"]
    headers = list(keyword_df["headers"].unique())
    keyword_df["required_kw"] = keyword_df.apply(
        lambda row: get_req_product_kw(row, keyword_df), axis=1
    )
    # load all csv into one df
    df = load_csv_into_df()

    # add keyword cols to df
    for header in headers:
        df[header] = 0

    # get lists of col headers, depending on generic or specific product
    generic = [header for header in headers if "generic" in header]
    non_generic = [header for header in headers if "generic" not in header]

    # check keywords of non-generic product column
    for header in non_generic:
        matched = keyword_df[keyword_df["headers"] == header]
        for row in matched.itertuples():
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
        for row in matched.itertuples():
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

    # pass chat data to llm
    for header in headers:
        df[header] = df[header].astype("object")
        df.loc[df[header] == 0, header] = ""
        # df[header] = df[header].apply(lambda row: pass_to_llm(row, header, keyword_df))

    # export df into xlsx
    with pd.ExcelWriter("./files/output.xlsx") as writer:
        df.to_excel(writer, sheet_name="processed", index=False)


if __name__ == "__main__":
    main()
