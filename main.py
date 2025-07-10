import pandas as pd
from pathlib import Path


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
        print("Processed", ", ".join(processed), end="\r")
    combined = pd.concat(df_list, ignore_index=True)
    return combined


# combine required product keyword
def get_req_product_kw(row: pd.Series, df: pd.DataFrame) -> str | None:
    if pd.isna(row["required_product"]):
        return None
    products = [p.strip() for p in row["required_product"].split(",")]
    keywords = df.loc[df["product"].isin(products), "keyword"].tolist()
    return "|".join(keywords)


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
        df[header] = ""
    
    # get lists of col headers, depending on generic or specific product
    generic = [header for header in headers if "generic" in header]
    non_generic = [header for header in headers if "generic" not in header]

    # check keywords of non-generic product column
    for header in non_generic:
        df[header] = 0
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
        df[header] = 0
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
            
    # export df into xlsx


if __name__ == "__main__":
    main()
