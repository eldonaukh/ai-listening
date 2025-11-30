import os
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

# Load environment variables once at module level
load_dotenv()

# =============================================================================
# Refactor Note: Abstract Base Class (Dependency Inversion Principle)
# Instead of a hardcoded 'match' statement in the old code, we define an interface.
# This allows us to swap 'Poe', 'DeepSeek', or 'GPT' without changing the main logic.
# =============================================================================
class LLMProvider(ABC):
    @abstractmethod
    def get_completion(self, messages: list) -> str:
        pass

# =============================================================================
# Refactor Note: Concrete Strategies (Open/Closed Principle)
# We can add new providers (e.g., Anthropic) by adding a class, 
# without modifying existing working code.
# =============================================================================
class DeepSeekClient(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), 
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"

    def get_completion(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            print(f"DeepSeek Error: {e}")
            return "{}" # Return empty JSON string on failure

class OpenAIClient(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def get_completion(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            print(f"OpenAI Error: {e}")
            return "{}"

class PoeClient(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("POE_API_KEY"), 
            base_url="https://api.poe.com/v1"
        )
        self.model = "Grok-4.1"

    def get_completion(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            print(f"Poe Error: {e}")
            return "{}"

# =============================================================================
# Refactor Note: Context Class (Single Responsibility Principle)
# This class is ONLY responsible for constructing the prompt and parsing the result.
# It doesn't care *which* AI provider is used.
# =============================================================================
class SentimentAnalyzer:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        # Refactor Note: The system prompt is now a class attribute, making it easier to manage.
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

    def analyze(self, user_prompt: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Delegate the API call to the provider
        response_str = self.provider.get_completion(messages)
        
        # Refactor Note: Error handling and JSON parsing are centralized here
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from AI")
            return {"sentiment": "Error", "reason": "JSON Parse Error"}

# Factory to easily switch providers based on string input (optional helper)
def get_analyzer(model_name: str = "gpt") -> SentimentAnalyzer:
    if model_name == "poe":
        return SentimentAnalyzer(PoeClient())
    elif model_name == "deepseek":
        return SentimentAnalyzer(DeepSeekClient())
    else:
        return SentimentAnalyzer(OpenAIClient())