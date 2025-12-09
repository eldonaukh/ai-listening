from utils.ai import get_analyzer
from utils.chatprocessor import ChatProcessor
from utils.preprocessor import Preprocessor


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
