import pandas as pd
from pathlib import Path
from tools.aho import matches_keyword_aho, build_automaton
from tools.ai import sentiment_check

KEYWORDS = {
    "abott": {"required": [], "include": ["abott", "雅培"], "exclude": []},
    "friso": {
        "required": [],
        "include": ["friso", "美素"],
        "exclude": ["gold", "prestige", "bio", "signature", "金裝", "皇家", "有機"],
    },
    "friso_gold": {
        "required": ["friso", "美素"],
        "include": ["gold", "金裝"],
        "exclude": [],
    },
}


def make_automation(keywords: dict):
    all_keywords = []
    for config in KEYWORDS.values():
        all_keywords.extend(config["required"] + config["include"] + config["exclude"])
    automaton = build_automaton(all_keywords)
    return automaton


def main():

    path = Path(".\\input_csv")
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
        # print(text_list)
        for text in text_list:
            count = 0
            answer = sentiment_check(text)
            mask.loc[index[count], "sentiment"] = answer
            count += 1
            if count == 10:
                break

        # print(mask.loc[count, "sentiment"])

        filename = ".\\output\\" + file.name
        mask.to_csv(filename)


if __name__ == "__main__":
    main()
