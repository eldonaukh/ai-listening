import pandas as pd
from pathlib import Path
from tools.aho import build_automaton
from tools.ai import sentiment_check
from typing import Any
from collections import defaultdict


def make_automation(keywords: dict):
    all_keywords = []
    for config in keywords.values():
        all_keywords.extend(config["required"] + config["include"] + config["exclude"])
    automaton = build_automaton(all_keywords)
    return automaton


def get_keywords(filepath: str = "./files/keywords.xlsx") -> Any:
    df = pd.read_excel("/files/keywords.xlsx/")
    return df.loc[
        (df["brand"] == "friso") & (df["product"] == "bio"),
        ["keyword", "required_product"],
    ]


def load_csv_into_df(path: str = "./files/chat/") -> pd.DataFrame | None:
    files = Path(path).iterdir()
    df_list = []
    for file in files:
        print("Processing", file, "...")

        try:
            df = pd.read_csv(file)
            df["Source"] = file.name
            df_list.append(df)
        except:
            print("File path error")
            return None
    combined = pd.concat(df_list, ignore_index=True)
    return combined


def get_col_headers(df: pd.DataFrame) -> list:
    df["headers"] = df["brand"] + "_" + df["product"]
    return list(df["headers"].unique())


def temp():
    path = Path("/files/chat/")
    files = path.iterdir()
    for file in files:
        print("Processing", file, "...")

        try:
            data = pd.read_csv(file)
        except:
            print("File path error")
            return
        mask = data.fillna(False)
        mask["bb"] = ""
        mask.loc[mask["MessageBody"].str.contains("bb", na=False), "bb"] = 1
        text_list = mask[mask["bb"] == 1]["MessageBody"]
        index = mask[mask["bb"] == 1]["MessageBody"].index
        for text in text_list:
            count = 0
            answer = sentiment_check(text)
            mask.loc[index[count], "sentiment"] = answer
            count += 1
            if count == 10:
                break

        filename = "/output/" + file.name
        mask.to_csv(filename)


def main():
    # get keywords from xlsx
    keyword_df = pd.read_excel("./files/keywords.xlsx")
    headers = get_col_headers(keyword_df)
    # load all csv into one df
    chat_df = load_csv_into_df()
    print(chat_df)
    # add keyword cols to df
    # tag message content contains keyword in brand cols
    # get keyword by col header
    # check keyword in message or not    #
    # export df into xlsx


if __name__ == "__main__":
    main()
