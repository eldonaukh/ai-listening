import pandas as pd
from utils.ai import SentimentAnalyzer
from utils.validator import KeywordSchema, ChatSchema, KeywordRow
from typing import cast
from pandera.typing import DataFrame
import json


class ChatProcessor:

    def __init__(
        self, keyword_df: DataFrame[KeywordSchema], analyzer: SentimentAnalyzer
    ):
        self._keyword_df = keyword_df
        self.analyzer = analyzer

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
        self.add_keywords_for_system_prompt()
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

            self._apply_mask(chat_df, g, skip_mask=skip_mask)
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
            keywords = self._get_keywords_for_prompt(header)

            mask = df[header] == 1

            if mask.any():
                print(df[mask].to_dict(orient="index"))

        return df

    def _pass_to_llm(self, row: pd.Series, header: str) -> pd.Series:
        data = f"Formula Brand: {header}, Message: {row["messageBody"]}"
        response = self.analyzer.analyze(data)
        row[header] = response["sentiment"]
        row["Reason"] = header + ": " + response["reason"] + "\n"
        return row

    def _get_keywords_for_prompt(self, header: str) -> str:
        keywords: list[str] = []
        matched = self._get_keyword_rows_of_header(header)

        for row in cast(list[KeywordRow], matched.itertuples(index=False)):
            if row.required_keyword:
                keywords.extend(row.required_keyword.split("|"))
            keywords.append(row.keyword)

        return ", ".join(set(keywords))

    def add_keywords_for_system_prompt(self) -> None:
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
