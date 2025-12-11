import argparse
import asyncio

from tqdm import tqdm

from utils.ai import get_analyzer
from utils.chatprocessor import ChatProcessor
from utils.merger import DataManager
from utils.preprocessor import Preprocessor


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chat Analysis and Processing Tool")

    # Path Arguments
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data",
        help="Base directory for data files (default: ./data)",
    )
    parser.add_argument(
        "--keyword_file",
        type=str,
        default="keywords.xlsx",
        help="Name of the keyword Excel file inside base_path (default: keywords.xlsx)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.xlsx",
        help="Name of the output Excel file inside base_path (default: output.xlsx)",
    )

    # Data Merge/Organization Arguments
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable the CSV merging and organization process",
    )
    parser.add_argument(
        "--merge_src",
        type=str,
        default="merge_src",
        help="Source folder for merging (relative to base_path)",
    )
    parser.add_argument(
        "--merge_dst",
        type=str,
        default="merge_dst",
        help="Destination folder for merging (relative to base_path)",
    )
    parser.add_argument(
        "--natures_dst",
        type=str,
        default="natures",
        help="Destination folder for organized natures (relative to base_path)",
    )
    parser.add_argument(
        "--group_info_file",
        type=str,
        default="group_info.csv",
        help="CSV file containing group nature info (relative to base_path)",
    )

    # AI/Analyzer Arguments
    parser.add_argument(
        "--provider",
        type=str,
        default="poe",
        help="AI Provider name (default: poe)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=400,
        help="Maximum concurrent AI tasks (default: 400)",
    )
    parser.add_argument(
        "--max_rate",
        type=int,
        default=400,
        help="Maximum request rate (default: 400)",
    )
    parser.add_argument(
        "--time_period",
        type=int,
        default=60,
        help="Time period for rate limiting in seconds (default: 60)",
    )

    return parser.parse_args()


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
    a = get_analyzer(
        "poe", "gemini-2.5-flash", max_concurrent_task=400, max_rate=400, time_preiod=60
    )
    if keyword is not None:
        c = ChatProcessor(keyword_df=keyword, analyzer=a)
        for sheet, chat in tqdm(chats.items(), desc=f"Processing chat data"):
            df = await c.process_chat_df(chat)
            chats[sheet] = df
            tqdm.write("Processed folder: " + sheet)
        c.save_result(chats, "./data/output.xlsx")


if __name__ == "__main__":
    asyncio.run(main())
