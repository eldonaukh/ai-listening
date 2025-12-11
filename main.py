import asyncio

from tqdm import tqdm

from utils.ai import get_analyzer
from utils.chatprocessor import ChatProcessor
from utils.merger import DataManager
from utils.preprocessor import Preprocessor


async def main() -> None:

    base_path = "./data"
    if False:
        d = DataManager(base_path)
        d.merge_csv_files(src="merge_src", dst="merge_dst")
        d.organize_csv_by_nature(
            src="merge_dst", dst="natures", group_nature="./data/group_info.csv"
        )

    p = Preprocessor(base_path)
    chats = p.get_chat_df_dict("natures")
    keyword = p.get_keyword_df("keywords.xlsx")
    a = get_analyzer("poe", "gemini-2.5-flash", max_concurrent_task=400, max_rate=400, time_preiod=60)
    if keyword is not None:
        c = ChatProcessor(keyword_df=keyword, analyzer=a)
        for sheet, chat in tqdm(chats.items(), desc=f"Processing chat data"):
            df = await c.process_chat_df(chat)
            chats[sheet] = df
            tqdm.write("Processed folder: " + sheet)
        c.save_result(chats, "./data/output.xlsx")


if __name__ == "__main__":
    asyncio.run(main())
