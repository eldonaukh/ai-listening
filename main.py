import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any

# Import the refactored classes
from tools.ai import get_analyzer, SentimentAnalyzer


# =============================================================================
# Refactor Note: DataLoader Class (Single Responsibility Principle)
# Handles all file reading/writing. Isolates I/O errors from business logic.
# =============================================================================
class DataLoader:
    def __init__(self, base_path: str = "./files"):
        self.base_path = Path(base_path)

    def load_keywords(self) -> pd.DataFrame:
        df = pd.read_excel(self.base_path / "keywords.xlsx")
        # Create headers immediately upon load
        df["headers"] = df["brand"] + "_" + df["product"]
        return df

    def load_chat_folder(self, folder_path: Path) -> pd.DataFrame:
        files = folder_path.iterdir()
        df_list = []
        for file in files:
            try:
                df = pd.read_csv(file)
                df["Source"] = file.name
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
                continue

        if not df_list:
            return pd.DataFrame()

        combined = pd.concat(df_list, ignore_index=True)

        # Ensure required columns exist, fill missing with None/NaN
        required_cols = [
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
            "Reason",
        ]
        return combined.reindex(columns=required_cols)

    def save_results(
        self, sheets: dict[str, pd.DataFrame], filename: str = "output.xlsx"
    ):
        output_path = self.base_path / filename
        with pd.ExcelWriter(output_path) as writer:
            for sheetname, dataframe in sheets.items():
                dataframe.to_excel(writer, sheet_name=sheetname, index=False)
                print(f"Sheet: {sheetname} added.")


# =============================================================================
# Refactor Note: ChatProcessor Class
# Encapsulates the complex logic of tagging keywords and calling the AI.
# =============================================================================
class ChatProcessor:
    def __init__(self, keyword_df: pd.DataFrame, analyzer: SentimentAnalyzer):
        self.keyword_df = keyword_df
        self.analyzer = analyzer

        # Pre-calculate required keywords to avoid doing it inside loops
        self.keyword_df["required_kw_processed"] = self.keyword_df.apply(
            lambda row: self._resolve_required_keywords(row), axis=1
        )
        self.headers = list(self.keyword_df["headers"].unique())

    def _resolve_required_keywords(self, row: pd.Series) -> str | None:
        """Helper to combine required product keywords (moved from global scope)."""
        if pd.isna(row["required_product"]):
            return None
        products = [p.strip() for p in row["required_product"].split(",")]
        keywords = self.keyword_df.loc[
            self.keyword_df["product"].isin(products), "keyword"
        ].tolist()
        return "|".join(keywords)

    def process_folder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrates the tagging and AI analysis."""
        df = df.copy()

        # 1. Initialize columns as Integers (0) for bitwise logic
        for header in self.headers:
            df[header] = 0

        # 2. Tag Keywords (This keeps columns as integers 0 or 1)
        self._tag_keywords(df)

        # 3. Type Conversion & Cleanup
        # We must convert int(0/1) to object/string BEFORE inserting text to avoid Dtype warnings
        for header in self.headers:
            # Explicitly cast to object to allow mixed strings/numbers temporarily
            df[header] = df[header].astype("object")

            # Replace 0 with empty string
            # Note: We compare against integer 0 because the values are still technically ints inside the object column
            df.loc[df[header] == 0, header] = ""

        # 4. Run AI Analysis
        self._run_sentiment_analysis(df)

        return df

    def _tag_keywords(self, df: pd.DataFrame):
        """
        Refactor Note: Logic Simplification
        Separates the identification of Generic vs Non-Generic headers
        but uses a shared helper method (_apply_mask) to avoid code duplication.
        """
        generic = [h for h in self.headers if "generic" in h]
        non_generic = [h for h in self.headers if "generic" not in h]

        # Tag Specific Brands first
        for header in non_generic:
            self._apply_mask(df, header, skip_mask=None)

        # Tag Generic Brands (Skipping rows where specific brands were found)
        for header in generic:
            main_brand = header.replace("_generic", "")
            subbrands = [b for b in non_generic if main_brand in b]

            # Logic: If any subbrand is found (value > 0), create a mask to skip
            # Note: In the old code, this relied on the previous loop having finished.
            mask_skip = df[subbrands].any(axis=1)
            self._apply_mask(df, header, skip_mask=mask_skip)

    def _apply_mask(self, df: pd.DataFrame, header: str, skip_mask: pd.Series | None):
        """
        Refactor Note: DRY (Don't Repeat Yourself)
        This helper replaces the copy-pasted loops in the old 'process_chat' function.
        It handles both the 'required keyword' check and the 'skip mask' check.
        """
        matched_rows = self.keyword_df[self.keyword_df["headers"] == header]

        for row in matched_rows.itertuples():
            # Determine the target rows (All rows OR Only non-skipped rows)
            target = (
                df["MessageBody"]
                if skip_mask is None
                else df.loc[~skip_mask, "MessageBody"]
            )

            # Check main keyword
            mask_kw = target.str.contains(row.keyword, case=False, na=False)

            # Check required keyword if it exists
            if row.required_kw_processed:
                mask_req = target.str.contains(
                    row.required_kw_processed, case=False, na=False
                )
                final_mask = mask_kw & mask_req
            else:
                final_mask = mask_kw

            # Apply back to DataFrame
            if skip_mask is None:
                df[header] = df[header] | final_mask.astype(int)
            else:
                df.loc[~skip_mask, header] = df.loc[
                    ~skip_mask, header
                ] | final_mask.astype(int)

    def _run_sentiment_analysis(self, df: pd.DataFrame):
        """
        Iterates through headers and calls the AI for tagged rows.
        """
        for header in self.headers:
            # Get keywords string for prompt
            kw_str = self._get_keywords_string(header)

            # Filter rows that need AI.
            # IMPORTANT: Since we converted to object, the '1' might be an integer 1
            # or a string "1" depending on pandas version behavior.
            # The safest check is to look for values that are NOT empty strings.
            mask = df[header] == 1

            if mask.any():
                # We use loc to ensure we are modifying the original dataframe
                # and pass the specific slice to the apply function
                df.loc[mask] = df.loc[mask].progress_apply(
                    lambda row: self._pass_to_llm(row, header, kw_str), axis=1
                )

    def _get_keywords_string(self, header: str) -> str:
        """Reconstructs the keyword string for the AI prompt."""
        keywords = []
        matched = self.keyword_df[self.keyword_df["headers"] == header]
        for row in matched.itertuples():
            if row.required_kw_processed:
                keywords.extend(row.required_kw_processed.split("|"))
            keywords.append(row.keyword)
        return ", ".join(set(keywords))

    def _pass_to_llm(self, row: pd.Series, header: str, keywords: str) -> pd.Series:
        """Prepares the prompt and updates the row with AI result."""
        prompt = f"Formula Brand: {header}, Keyword: {keywords}, Message: {row['MessageBody']}"

        result = self.analyzer.analyze(prompt)

        # Update row
        row[header] = result.get("sentiment", "")
        row["Reason"] = result.get("reason", "")
        return row


# =============================================================================
# Refactor Note: Main Execution
# Clean entry point. Orchestrates the components defined above.
# =============================================================================
def main():
    tqdm.pandas()

    # 1. Setup
    loader = DataLoader("./files")

    try:
        keyword_df = loader.load_keywords()
    except Exception as e:
        print(f"Failed to load keywords: {e}")
        return

    # Initialize AI (Strategy Pattern in action - easy to switch 'poe' to 'deepseek')
    analyzer = get_analyzer("poe")
    processor = ChatProcessor(keyword_df, analyzer)

    sheets = {}
    chat_path = Path("./files/chat")

    # 2. Processing Loop
    if not chat_path.exists():
        print("Chat folder not found.")
        return

    subfolders = [f for f in chat_path.iterdir() if f.is_dir()]

    for sub in subfolders:
        print(f"Start processing: {sub.name}")

        df = loader.load_chat_folder(sub)
        if df.empty:
            print(f"No data in {sub.name}")
            continue

        processed_df = processor.process_folder(df)
        sheets[sub.name] = processed_df
        print(f"Processed: {sub.name}")

    # 3. Export
    loader.save_results(sheets)


if __name__ == "__main__":
    main()
