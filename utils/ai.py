import os
import json
import asyncio
import re
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI, APIStatusError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from pydantic import ValidationError
from utils.validator import SentimentResponse

load_dotenv()


class LLMProvider:

    def __init__(self, name: str, base_url: str, model: str):
        self.name = name
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.client_async = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def api_key(self) -> str:
        api_key_name = f"{self.name.upper()}_API_KEY"
        api_key = os.getenv(api_key_name)
        if isinstance(api_key, str):
            return api_key
        else:
            return "error"

    def get_completion(
        self, messages: list[ChatCompletionMessageParam], model: str
    ) -> tuple[Literal[True], ChatCompletion] | tuple[Literal[False], str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            return True, response
        except APIStatusError as e:
            return False, f"APIStatusError: {e}"

    async def get_completion_async(
        self, messages: list[ChatCompletionMessageParam], model: str
    ) -> tuple[Literal[True], ChatCompletion] | tuple[Literal[False], str]:
        try:
            response = await self.client_async.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            return True, response
        except APIStatusError as e:
            return False, f"APIStatusError: {e}"


class SentimentAnalyzer:

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.system_prompt = """
### 角色設定
你是一位專精於嬰兒配方奶粉及母嬰健康的市場研究分析師。你的任務是分析媽媽群組（WhatsApp）對話中的情緒。

### 任務
分析用戶提供的文本（WhatsApp 對話記錄），針對「特定奶粉品牌」進行情緒分析。在分析時，請特別注意參考提供的【關鍵字定義】。

### 規則
1. **僅限目標品牌：** 只專注於針對該特定品牌的情緒。忽略對競爭對手的評論，除非該評論直接影響目標品牌的觀感。
2. **關鍵字匹配：** 請檢查文本是否包含【關鍵字定義】中的詞彙。
   - 如果包含正面關鍵字，傾向於判斷為 P。
   - 如果包含負面關鍵字，傾向於判斷為 N。
3. **情緒判斷邏輯：**
   - **P (正面)：** 讚賞、推薦、有意購買、提及正面健康效果（如：長肉、大便靚），或命中正面關鍵字。
   - **N (負面)：** 投訴、副作用（如：便秘、熱氣、敏感）、價格過高、拒絕購買，或命中負面關鍵字。
   - **I (中立)：** 一般查詢（如：哪裡買？）、事實陳述、情緒好壞參半、提及品牌但無主觀評價。
4. **輸出格式：** 僅回傳一個原始 JSON 物件。不要使用 Markdown 格式（如 ```json）。
5. **語言：** JSON 中的 `reason` 欄位必須使用繁體中文。

### JSON 結構
{
    "sentiment": "P", // 或 "N", 或 "I"
    "reason": "在此輸入50字以內的繁體中文解釋，說明判斷原因（若有命中關鍵字請提及）"
}
"""

    def system_prompt_insert_keywords(self, keywords: str):
        updated_prompt = f"""
        ### 角色設定
你是一位專精於嬰兒配方奶粉及母嬰健康的市場研究分析師。你的任務是分析媽媽群組（WhatsApp）對話中的情緒。

### 任務
分析用戶提供的文本（WhatsApp 對話記錄），針對「特定奶粉品牌」進行情緒分析。在分析時，請特別注意參考提供的【關鍵字定義】。

### 品牌關鍵字定義 (JSON)
以下是用於輔助判斷評論相關品牌的關鍵字列表：
{keywords}

### 規則
1. **僅限目標品牌：** 只專注於針對該特定品牌的情緒。
2. **關鍵字匹配：** 請檢查文本是否包含【品牌關鍵字定義】中的詞彙。
   - 評論如果出現關鍵字，可傾向於判斷評論為關鍵字相關品牌。
3. **情緒判斷邏輯：**
   - **P (正面)：** 讚賞、推薦、有意購買、提及正面健康效果（如：長肉、大便靚），或命中正面關鍵字。
   - **N (負面)：** 投訴、副作用（如：便秘、熱氣、敏感）、價格過高、拒絕購買，或命中負面關鍵字。
   - **I (中立)：** 一般查詢（如：哪裡買？）、事實陳述、情緒好壞參半、提及品牌但無主觀評價。
4. **輸出格式：** 僅回傳一個原始 JSON 物件。嚴格遵守下方輸出 JSON 結構，不可有任何格式以外文字。
5. **語言：** JSON 中的 `reason` 欄位必須使用繁體中文。

### 輸出 JSON 結構
{{
    "sentiment": "P", 或 "N", 或 "I"
    "reason": "在此輸入50字以內的繁體中文解釋，說明判斷原因（若有命中關鍵字請提及）"
}}
        """
        self.system_prompt = updated_prompt

    async def analyze(self, user_prompt: str, index: int, header: str) -> SentimentResponse:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self.get_valid_response(messages, SentimentResponse)

    async def get_valid_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model_class: type[SentimentResponse],
        max_retries: int = 3,
    ):

        for _ in range(max_retries):
            success, response = await self.provider.get_completion_async(
                messages, self.provider.model
            )

            try:
                if isinstance(response, ChatCompletion) and isinstance(
                    response.choices[0].message.content, str
                ):
                    response_text: str = response.choices[0].message.content
                    return self.parse_and_validate(response_text, model_class)
            except ValueError as e:
                # Feed the error back to the LLM!
                error_msg = f"Your JSON was invalid. Error: {str(e)}. Please fix and output only JSON."
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": error_msg})

        raise Exception(
            f"Failed to get valid JSON after {max_retries} times of retries"
        )

    def parse_and_validate(
        self, llm_output: str, model_class: type[SentimentResponse]
    ) -> SentimentResponse:
        """
        Strips markdown, parses JSON, and validates against Pydantic model.
        """
        try:
            # Step A: Clean Markdown (LLMs love to wrap JSON in ```json ... ```)
            # Look for content between ```json and ``` or just ``` and ```
            match = re.search(r"```(?:json)?(.*?)```", llm_output, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # Assume the whole string is JSON if no markdown found
                json_str = llm_output.strip()

            # Step B: Parse JSON
            data = json.loads(json_str)

            # Step C: Validate against Pydantic
            validated_obj = model_class.model_validate(data)
            return validated_obj

        except json.JSONDecodeError:
            raise ValueError("LLM did not return valid JSON syntax.")
        except ValidationError as e:
            raise ValueError(f"JSON structure invalid: {e}")


def get_analyzer(provider_name: str, model_name: str) -> SentimentAnalyzer:
    provider_name = provider_name.lower().strip()
    match provider_name.lower().strip():
        case "poe":
            base_url = "https://api.poe.com/v1"

    provider = LLMProvider(provider_name, base_url, model_name)
    return SentimentAnalyzer(provider)
