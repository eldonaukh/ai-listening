import sys
import ctypes
import os
import asyncio
from tqdm import tqdm

# Import Gooey
from gooey import Gooey, GooeyParser

# Import your existing modules
from utils.merger import DataManager
from utils.preprocessor import Preprocessor
from utils.chatprocessor import ChatProcessor
from utils.ai import get_analyzer

# Ensure the event loop policy is set for Windows if needed
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        # Fallback for older Windows versions (Win 7/8)
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


@Gooey(
    program_name="Chat Sentiment Analyzer",
    program_description="Process chat logs, merge data, and perform AI sentiment analysis.",
    default_size=(900, 700),
    navigation="TABBED",  # Tabs make the UI cleaner
    # progress_regex=r"^Progress: (\d+)%$",  # Regex to catch progress updates
    # progress_expr="x[0]",
    timing_options={"show_time_remaining": True},
    header_bg_color="#ABCEFF",
    body_bg_color="#F0F0F0",
    terminal_font_family="Consolas",
)
def main():
    parser = GooeyParser(description="Argument Parser for Chat Processing")

    top_group = parser.add_argument_group(
        "Base File Path",
        "Configure the base path of all file / folder paths of options below",
    )

    top_group.add_argument(
        "--base_path",
        type=str,
        default="./data",
        help="Base directory for data storage",
        widget="DirChooser",  # Adds a folder selection button
    )

    # --- Tab 1: File Paths & Directories ---
    merger_group = parser.add_argument_group(
        "CSV Merger: merge chat CSV files in subfolders",
        "Configure input and output directories of CSV merging operation",
    )

    merger_group.add_argument(
        "--merge_src",
        type=str,
        default="./data/merge_src",
        help="Subfolder name containing raw CSV files to merge",
        widget="DirChooser",
    )

    merger_group.add_argument(
        "--merge_dst",
        type=str,
        default="./data/merge_dst",
        help="Subfolder name to save merged CSV files",
        widget="DirChooser",
    )

    # --- Tab 2: File Paths & Directories ---
    organizer_group = parser.add_argument_group(
        "CSV Organizer: organize chat CSV files by group natures in reference file",
        "Configure reference file and output directories of CSV by group natures",
    )

    organizer_group.add_argument(
        "--group_info_file",
        type=str,
        default="./data/group_info.csv",
        help="Path to the CSV / Excel file containing group info and nature",
        widget="FileChooser",
    )

    organizer_group.add_argument(
        "--natures_dst",
        type=str,
        default="./data/chats",
        help="Subfolder name to save CSVs organized by nature/group",
        widget="DirChooser",
    )

    # --- Tab 3: AI & Processing Settings ---
    ai_group = parser.add_argument_group(
        "Chat Processer: tag brand keywords and analyze message sentiment with power of AI",
        "Settings for keyword reference file and LLM model",
    )

    ai_group.add_argument(
        "--keyword_file",
        type=str,
        default="./data/keywords.xlsx",
        help="Path to the Excel file containing brand keywords",
        widget="FileChooser",  # Adds a file selection button
    )

    ai_group.add_argument(
        "--output_file",
        type=str,
        default="./data/final_analysis.xlsx",
        help="Filename for the final Excel output",
        widget="FileSaver",
    )

    ai_group.add_argument(
        "--provider",
        type=str,
        choices=["poe"],
        default="poe",
        help="LLM Provider to use",
        widget="Dropdown",
    )

    ai_group.add_argument(
        "--model",
        type=str,
        choices=["gemini-2.5-flash", "gpt-4.1-nano", "gpt-5-nano", "claude-haiku-3.5"],
        default="gemini-2.5-flash",
        help="Model name (e.g., gpt-4, gpt-3.5-turbo)",
        widget="Dropdown",
    )

    ai_group.add_argument(
        "--max_concurrent",
        type=int,
        default=400,
        help="Maximum concurrent API requests",
    )

    ai_group.add_argument(
        "--max_rate", type=int, default=400, help="Maximum requests per time period"
    )

    ai_group.add_argument(
        "--time_period",
        type=int,
        default=60,
        help="Time period for rate limiting (in seconds)",
    )

    ai_group.add_argument(
        "--message_col",
        type=str,
        default="messageBody",
        help="Column name in CSV containing the message text",
    )

    args = parser.parse_args()

    # Run the async logic
    try:
        asyncio.run(run_processing(args))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nCritical Error: {e}")
        # In Gooey, raising the error keeps the window open so user can see it
        raise e


async def run_processing(args):
    """
    The main async logic, separated from the Gooey definition.
    """
    print("--- Starting Chat Processing ---")

    # 1. Initialize DataManager
    print("Initializing Data Manager...")
    dm = DataManager(base_path=args.base_path)

    # 2. Merge Files
    print(f"Merging files from {args.merge_src} to {args.merge_dst}...")
    dm.merge_csv_files(src=args.merge_src, dst=args.merge_dst)

    # 3. Organize by Nature
    print(f"Organizing by nature into {args.natures_dst}...")
    dm.organize_csv_by_nature(
        src=args.merge_dst, dst=args.natures_dst, group_nature=args.group_info_file
    )

    # 4. Initialize Preprocessor
    print("Initializing Preprocessor...")
    pre = Preprocessor(base_path=args.base_path)

    # 5. Load Keywords
    print(f"Loading keywords from {args.keyword_file}...")
    keyword_df = pre.get_keyword_df(file_path=args.keyword_file)

    # 6. Load Chat Dataframes
    print("Loading chat dataframes...")
    chats = pre.get_chat_df_dict(chat_folder=args.natures_dst)

    if not chats:
        print("No chat files found to process.")
        return

    # 7. Initialize AI Analyzer
    print(f"Initializing AI Analyzer ({args.provider} - {args.model})...")
    analyzer = get_analyzer(
        provider_name=args.provider,
        model_name=args.model,
        max_concurrent_task=args.max_concurrent,
        max_rate=args.max_rate,
        time_period=args.time_period,
    )

    # 8. Initialize ChatProcessor
    if keyword_df is not None:
        c = ChatProcessor(keyword_df=keyword_df, analyzer=analyzer)

        # 9. Process Chats
        print(f"Processing {len(chats)} chat groups...")

        processed_dfs = []

        # We use a manual counter for Gooey progress bar compatibility
        total_items = len(chats)
        current_item = 0

        for sheet, chat in chats.items():
            print(f"Processing sheet: {sheet}")

            # Note: Ensure you updated ChatProcessor to accept message_col if you added that arg
            # If not, remove `message_col=args.message_col` below
            df = await c.process_chat_df(chat)

            processed_dfs.append(df)

            # Update Progress for Gooey
            current_item += 1
            progress_percent = int((current_item / total_items) * 100)
            print(f"Progress: {progress_percent}%")
            sys.stdout.flush()  # Ensure Gooey catches the print immediately

        # 10. Save Final Result
        final_path = os.path.join(args.base_path, args.output_file)
        print(f"Saving final merged analysis to {final_path}...")

        if processed_dfs:
            # Concatenate all processed dataframes
            import pandas as pd

            final_df = pd.concat(processed_dfs, ignore_index=True)
            final_df.to_excel(final_path, index=False)
            print("Success! Processing complete.")
        else:
            print("Warning: No data was processed.")


async def manual() -> None:

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
        "poe", "gemini-2.5-flash", max_concurrent_task=400, max_rate=400, time_period=60
    )
    if keyword is not None:
        c = ChatProcessor(keyword_df=keyword, analyzer=a)
        for sheet, chat in tqdm(chats.items(), desc=f"Processing chat data"):
            df = await c.process_chat_df(chat)
            chats[sheet] = df
            tqdm.write("Processed folder: " + sheet)
        c.save_result(chats, "./data/output.xlsx")


if __name__ == "__main__":
    # Gooey requires the script to be run directly
    main()
