import asyncio
import json
import re
from typing import Any, Coroutine, cast

import pandas as pd
from pandera.typing import DataFrame
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdmas

from utils.ai import SentimentAnalyzer
from utils.validator import (
    ChatRow,
    ChatSchema,
    KeywordRow,
    KeywordSchema,
    SentimentResponse,
)


class ChatProcessor:

    def __init__(
        self, keyword_df: DataFrame[KeywordSchema], analyzer: SentimentAnalyzer
    ):
        self._keyword_df = keyword_df
        self.analyzer = analyzer
        self.sem = asyncio.Semaphore(2)

    @property
    def keyword_df(self):
        return self._keyword_df

    @property
    def unique_headers(self) -> list[str]:
        return list(self._keyword_df["headers"].unique())

    @property
    def generic_headers(self) -> list[str]:
        return [header for header in self.unique_headers if "generic" in header]

    @property
    def non_generic_headers(self) -> list[str]:
        return [header for header in self.unique_headers if "generic" not in header]

    def process_chat_df(self, chat_df: DataFrame[ChatSchema]) -> DataFrame[ChatSchema]:
        df = self._add_header_columns_to_chat_df(chat_df)
        df = self._tag_keywords(df)
        self._add_keywords_for_system_prompt()
        df = self._check_sentiment(df)

        return df

    def _get_keyword_rows_of_header(self, header: str) -> DataFrame[KeywordSchema]:
        return self._keyword_df[self._keyword_df["headers"] == header]

    def _tag_keywords(self, chat_df: DataFrame[ChatSchema]) -> DataFrame[ChatSchema]:
        for ng in self.non_generic_headers:
            self._apply_mask(chat_df, ng, skip_mask=None)

        for g in self.generic_headers:
            main = g.replace("_generic", "")
            subbrand = [brand for brand in self.non_generic_headers if main in brand]
            skip_mask = chat_df[subbrand].any(axis=1)

            self._apply_mask_vec(chat_df, g, skip_mask=skip_mask)
        return chat_df

    def _apply_mask(
        self,
        chat_df: DataFrame[ChatSchema],
        header: str,
        skip_mask: pd.Series | None,
        message_column: str = "messageBody",
    ) -> DataFrame[ChatSchema]:
        matched = self._get_keyword_rows_of_header(header)

        target = (
            chat_df[message_column]
            if skip_mask is None
            else chat_df.loc[~skip_mask, message_column]
        )
        for row in cast(list[KeywordRow], matched.itertuples(index=False)):
            mask_keyword = target.str.contains(row.keyword, case=False, na=False)
            if row.required_keyword:
                mask_required = target.str.contains(
                    row.required_keyword, case=False, na=False
                )
                final_mask = mask_required & mask_keyword
            else:
                final_mask = mask_keyword

        if skip_mask is not None:
            chat_df.loc[~skip_mask, header] = chat_df.loc[
                ~skip_mask, header
            ] | final_mask.astype(int)
        else:
            chat_df[header] = chat_df[header] | final_mask.astype(int)

        return chat_df


    def _apply_mask_vec(
        self,
        chat_df: pd.DataFrame,  # simplified type hint for clarity
        header: str,
        skip_mask: pd.Series | None,
        message_column: str = "messageBody",
    ) -> pd.DataFrame:
        matched = self._get_keyword_rows_of_header(header)

        # If no keywords match this header, return early to save processing
        if matched.empty:
            return chat_df

        # 1. Define the target series based on the skip_mask
        if skip_mask is None:
            target_indices = chat_df.index
            target_series = chat_df[message_column]
        else:
            target_indices = chat_df.index[~skip_mask]
            target_series = chat_df.loc[~skip_mask, message_column]

        # 2. Construct a single compiled Regex pattern
        # The original logic seemed to imply OR logic between rows (though the loop overwrote it).
        # We assume you want to flag the row if ANY of the keyword rules match.

        # Group 1: Keywords that have NO required_keyword
        simple_keywords = matched[
            matched["required_keyword"].isna() | (matched["required_keyword"] == "")
        ]["keyword"].tolist()

        # Group 2: Keywords that DO have a required_keyword
        complex_rows = matched[
            matched["required_keyword"].notna() & (matched["required_keyword"] != "")
        ]

        # Initialize a mask of False for all target rows
        combined_mask = pd.Series(False, index=target_indices)

        # 3. Vectorized application for Simple Keywords (OR logic)
        if simple_keywords:
            # Escape keywords to treat them as literal strings in regex, join with OR pipe
            pattern = "|".join(map(re.escape, simple_keywords))
            combined_mask |= target_series.str.contains(pattern, case=False, na=False)

        # 4. Vectorized application for Complex Keywords (AND logic per row)
        # We cannot easily join these into one regex because of the AND relationship (keyword + required).
        # However, we can iterate ONLY the complex rules (usually fewer) or construct a specific lookahead regex.
        # Approach: Iterate complex rows (still faster than row-by-row if few complex rules exist)
        if not complex_rows.empty:
            for row in cast(list, complex_rows.itertuples(index=False)):
                # Check (Contains Keyword) AND (Contains Required)
                row_mask = target_series.str.contains(
                    row.keyword, case=False, na=False
                ) & target_series.str.contains(row.required_keyword, case=False, na=False)
                combined_mask |= row_mask

        # 5. Apply the result back to the DataFrame
        # Ensure the column exists and is integer type if not already
        if header not in chat_df.columns:
            chat_df[header] = 0

        # We use bitwise OR to merge with existing flags in that column
        chat_df.loc[target_indices, header] = chat_df.loc[
            target_indices, header
        ] | combined_mask.astype(int)

        return chat_df

    def _add_header_columns_to_chat_df(
        self, chat_df: DataFrame[ChatSchema]
    ) -> DataFrame[ChatSchema]:
        for header in self.unique_headers:
            chat_df[header] = 0
        return chat_df

    def _chat_df_zero_to_string(
        self, chat_df: DataFrame[ChatSchema], header: str
    ) -> DataFrame[ChatSchema]:
        # turn 0 into empty str
        chat_df[header] = chat_df[header].astype("object")
        chat_df.loc[chat_df[header] == 0, header] = ""
        return chat_df

    def _check_sentiment(self, chat_df: DataFrame[ChatSchema]):
        for header in self.unique_headers:
            df = self._chat_df_zero_to_string(chat_df, header)
            # keywords = self._get_keywords_for_prompt(header)
            tasks: list[asyncio.Task[tuple[str, int, SentimentResponse]]] = []
            task_list: list[Coroutine[Any, Any, tuple[str, int, SentimentResponse]]] = (
                []
            )
            # loop = asyncio.get_event_loop()
            mask = df[header] == 1

            if mask.any():
                for row in cast(
                    list[ChatRow],
                    tqdm(
                        df[mask].itertuples(),
                        total=df[mask].shape[0],
                        desc=f"Creating tasks: {header}",
                    ),
                ):
                    user_prompt = f"Formula Brand: {header}, Message: {row.messageBody}"
                    # task = loop.create_task(self._wrap_analyze_with_index(user_prompt, int(row.Index), header))
                    # tasks.append(task)
                    task_list.append(
                        self._wrap_analyze_with_index(
                            user_prompt, int(row.Index), header
                        )
                    )

                results = asyncio.run(self._run_async_check(task_list))

                for result in results:
                    header, index, response = result
                    # index_str = str(index)

                    if response.success:
                        df.loc[index, header] = response.sentiment

                    current_reason = str(df.loc[index, "Reason"])
                    df.loc[index, "Reason"] = (
                        current_reason + f"{header}: {response.reason}\n"
                    )
        return df

    async def _run_async_check(
        self, tasks: list[Coroutine[Any, Any, tuple[str, int, SentimentResponse]]]
    ):
        async with self.sem:
            responses: list[tuple[str, int, SentimentResponse]] = await tqdmas.gather(
                *tasks
            )
            # print(f"Completed {len(responses)} requests")
            return responses

    async def _wrap_analyze_with_index(
        self, user_prompt: str, index: int, header: str
    ) -> tuple[str, int, SentimentResponse]:
        # Add a small sleep to prevent "bursts" even within the semaphore limit
        await asyncio.sleep(1)
        try:
            response = await self.analyzer.analyze(user_prompt)
            return header, index, response
        except Exception as e:
            return (
                header,
                index,
                SentimentResponse(success=False, sentiment="I", reason=str(e)),
            )

    # def _pass_to_llm_apply(self, row: pd.Series, header: str) -> pd.Series:
    #     data = f"Formula Brand: {header}, Message: {row["messageBody"]}"
    #     response = self.analyzer.analyze(data)
    #     row[header] = response.sentiment
    #     row["Reason"] = header + ": " + response.reason + "\n"
    #     return row

    def _get_keywords_for_prompt(self, header: str) -> str:
        keywords: list[str] = []
        matched = self._get_keyword_rows_of_header(header)

        for row in cast(list[KeywordRow], matched.itertuples(index=False)):
            if row.required_keyword:
                keywords.extend(row.required_keyword.split("|"))
            keywords.append(row.keyword)

        return ", ".join(set(keywords))

    def _add_keywords_for_system_prompt(self) -> None:
        keyword_dict = self.keyword_df.to_dict(orient="records")
        self.analyzer.system_prompt_insert_keywords(json.dumps(keyword_dict))

    def save_result(
        self, dataframes: dict[str, DataFrame[ChatSchema]], output_path: str
    ):
        with pd.ExcelWriter(output_path) as writer:
            for sheetname, dataframe in dataframes.items():
                dataframe.to_excel(writer, sheet_name=sheetname, index=False)
                print(f"Sheet: {sheetname} has been added to file.")
        print("Output of file has been done.")
