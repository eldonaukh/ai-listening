import os
import json
import httpx
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()


class LLMProvider:

    def __init__(self, name: str, base_url: str, model: str):
        self.name = name
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
        except APIStatusError as e:
            return "{}"

        if isinstance(response.choices[0].message.content, str):
            return response.choices[0].message.content
        else:
            return "{}"


class SentimentAnalyzer:

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.system_prompt = """
        你是孕婦健康及初生嬰兒育兒專家、嬰兒奶粉品牌專家, 永遠以JSON格式回應。
        用戶會提供嬰兒奶粉品牌名稱, 品牌相關關鍵字, 及奶粉品牌相關的 WhatsApp 媽媽群組對話給你, 
        請判斷用戶提供的文本對提及的品牌情緒是正面(P)、負面(N)、中立(I), 並提供情緒判斷的原因。
        必須使用以下JSON格式回應:
        {
            "sentiment": "sentiment code, either 'P', 'N' or 'I'",
            "reason": "reason of sentiment within 50 traditional Chinese characters"
        }
        Do not include any text outside the JSON object. Strictly adhere to this format.    
        """

    def analyze(self, user_prompt: str) -> dict[str, str]:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_str = self.provider.get_completion(messages, self.provider.model)

        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from AI")
            return {"sentiment": "Error", "reason": "JSON Parse Error"}


def get_analyzer(provider_name: str, model_name: str) -> SentimentAnalyzer:
    provider_name = provider_name.lower().strip()
    match provider_name.lower().strip():
        case "poe":
            base_url = "https://api.poe.com/v1"

    provider = LLMProvider(provider_name, base_url, model_name)
    return SentimentAnalyzer(provider)
