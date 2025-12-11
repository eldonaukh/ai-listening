import shutil
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm


class DataManager:

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    def get_filenames(
        self,
        src_name: str,
        filename: str,
        file_type: str = "csv",
    ):
        src_path = self.base_path / src_name
        dst_path = self.base_path / filename
        groups = list(file.name.replace(".csv", "") for file in src_path.rglob("*.csv"))
        df = pd.DataFrame({"gus_id": groups})
        df["group_nature"] = None
        if file_type.lower().strip() == "xlsx":
            df.to_excel(dst_path, index=False)
        else:
            df.to_csv(dst_path, index=False)

    def merge_csv_files(
        self,
        src: str = "merge_src",
        dst: str = "merge_dst",
    ):
        """
        Scans subfolders in src, merges CSVs with the same filename,
        sorts by Date2/Time, removes duplicates, and saves to dst.
        """

        # 1. Setup Paths
        src_path = self.base_path / src
        dst_path = self.base_path / dst

        if not src_path.exists():
            print(f"Error: Input directory '{src}' not found.")
            return

        # Create output directory if it doesn't exist
        dst_path.mkdir(exist_ok=True)
        print(f"Processing files from '{src}' into '{dst}'...")

        # 2. Group files by filename
        # Dictionary structure: { 'filename.csv': [Path_obj_1, Path_obj_2, ...] }
        files_map: dict[str, list[Path]] = {}

        # Walk through all subfolders
        all_files = list(src_path.rglob("*.csv"))

        if not all_files:
            print("No CSV files found in subfolders.")
            return

        print(f"Found {len(all_files)} total CSV files. Grouping by filename...")

        for file_path in all_files:
            if file_path.is_file():
                filename = file_path.name
                if filename not in files_map:
                    files_map[filename] = []
                files_map[filename].append(file_path)

        # 3. Process each unique filename
        for filename, file_paths in tqdm(files_map.items(), desc="Merging Files"):

            try:
                # List to hold dataframes for this specific filename
                dfs = []

                # Read all files with this name
                for fp in file_paths:
                    try:
                        # Read CSV.
                        # We read everything as strings initially to ensure we don't lose leading zeros
                        # or have pandas guess types incorrectly before merging.
                        df = pd.read_csv(fp, dtype=str)
                        dfs.append(df)
                    except pd.errors.EmptyDataError:
                        tqdm.write(f"Warning: Skipped empty file {fp}")
                    except Exception as e:
                        tqdm.write(f"Error reading {fp}: {e}")

                if not dfs:
                    continue

                # Merge all dataframes for this filename
                merged_df = pd.concat(dfs, ignore_index=True)

                # 4. Remove Duplicates
                # Keeps the first occurrence, drops subsequent identical rows
                merged_df.drop_duplicates(inplace=True)

                # 5. Sort by 'Date2' and 'Time'
                # Check if columns exist before sorting
                if "Date2" in merged_df.columns and "Time" in merged_df.columns:

                    # --- FIX APPLIED HERE ---
                    # Added dayfirst=True to handle dd/mm/yyyy format correctly
                    merged_df["temp_sort_date"] = pd.to_datetime(
                        merged_df["Date2"], dayfirst=True, errors="coerce"
                    )

                    # Handle Time parsing
                    merged_df["temp_sort_time"] = pd.to_datetime(
                        merged_df["Time"], format="%H:%M:%S", errors="coerce"
                    ).dt.time

                    # Sort
                    # We sort by the temp columns if created
                    if "temp_sort_date" in merged_df.columns:
                        merged_df.sort_values(
                            by=["temp_sort_date", "temp_sort_time"],
                            ascending=[True, True],
                            inplace=True,
                        )
                        # Drop temp columns so they don't appear in the final file
                        merged_df.drop(
                            columns=["temp_sort_date", "temp_sort_time"], inplace=True
                        )
                    else:
                        # Fallback to string sorting if datetime conversion failed completely
                        merged_df.sort_values(
                            by=["Date2", "Time"], ascending=[True, True], inplace=True
                        )
                else:
                    tqdm.write(
                        f"Notice: '{filename}' missing 'Date2' or 'Time' columns. Saved without specific sort."
                    )

                # 6. Save to 'merged' folder
                output_file_path = dst_path / filename
                merged_df.to_csv(output_file_path, index=False)

            except Exception as e:
                tqdm.write(f"Failed to process group '{filename}': {e}")

        print("Merge complete.")

    def organize_csv_by_nature(
        self,
        src: str,
        dst: str,
        group_nature: str,
    ):

        src_path = self.base_path / src
        dst_path = self.base_path / dst

        files = list(src_path.rglob("*.csv"))

        if group_nature.endswith(".csv"):
            nature_df = pd.read_csv(group_nature, dtype=str)
        elif group_nature.endswith(".xlsx"):
            nature_df = pd.read_excel(group_nature, dtype=str)

        nature_df["filename"] = nature_df["gus_id"] + ".csv"
        nature_dict = cast(list[dict[str, str]], nature_df.to_dict(orient="records"))

        # create folders by nature
        natures: list[str] = list(nature_df["group_nature"].astype(str).unique())
        nature_paths = {}
        for nature in natures:
            folder = dst_path / nature
            print(dst_path)
            print(folder)
            folder.mkdir(exist_ok=True, parents=True)
            nature_paths[nature] = dst_path / nature

        for file in files:
            nature = next(
                (
                    item["group_nature"]
                    for item in nature_dict
                    if item["filename"] == file.name
                ),
                "",
            )
            if len(nature) > 1:
                dest = dst_path / nature
                shutil.copy2(file, dest)
