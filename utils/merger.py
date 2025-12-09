import shutil
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm


def merge_csv_files(
    base_name: str = "../data/",
    source_name: str = "to_merge",
    dest_name: str = "merged",
):
    """
    Scans subfolders in source_name, merges CSVs with the same filename,
    sorts by Date2/Time, removes duplicates, and saves to dest_name.
    """

    # 1. Setup Paths
    base_path = Path(base_name)
    source_path = base_path / source_name
    dest_path = base_path / dest_name

    if not source_path.exists():
        print(f"Error: Input directory '{source_name}' not found.")
        return

    # Create output directory if it doesn't exist
    dest_path.mkdir(exist_ok=True)
    print(f"Processing files from '{source_name}' into '{dest_name}'...")

    # 2. Group files by filename
    # Dictionary structure: { 'filename.csv': [Path_obj_1, Path_obj_2, ...] }
    files_map: dict[str, list[Path]] = {}

    # Walk through all subfolders
    all_files = list(source_path.rglob("*.csv"))

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
            output_file_path = dest_path / filename
            merged_df.to_csv(output_file_path, index=False)

        except Exception as e:
            tqdm.write(f"Failed to process group '{filename}': {e}")

    print("Merge complete.")


def get_filenames_to_csv(
    folder: str = "../data/merged/", output: str = "../data/merged_groups.csv"
):
    base_path = Path(folder)
    groups = list(file.name.replace(".csv", "") for file in base_path.rglob("*.csv"))
    df = pd.DataFrame({"gus_id": groups})
    df["group_nature"] = None
    df.to_csv(output, index=False)


def organize_csv_by_nature(
    base_path_name: str = "../data/",
    source_name="merged",
    dest_name="natures",
    group_nature: str = "../data/merged_groups.csv",
):

    base_path = Path(base_path_name)
    source_path = base_path / source_name
    dest_path = base_path / dest_name

    files = list(source_path.rglob("*.csv"))
    nature_df = pd.read_csv(group_nature, dtype=str)
    nature_df["filename"] = nature_df["gus_id"] + ".csv"
    nature_dict = cast(list[dict[str, str]], nature_df.to_dict(orient="records"))

    # create folders by nature
    natures: list[str] = list(nature_df["group_nature"].astype(str).unique())
    nature_paths = {}
    for nature in natures:
        folder = dest_path / nature
        folder.mkdir()
        nature_paths[nature] = dest_path / nature

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
            dest = dest_path / nature
            shutil.copy2(file, dest)


def main():
    organize_csv_by_nature()


if __name__ == "__main__":
    main()
