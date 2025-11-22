import os
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError
from openai.types.chat import ChatCompletionMessageParam

def sentiment_check(user_prompt: str, model_to_ask: str = "poe"):
    load_dotenv()
    match model_to_ask:
        case "poe":
            model = "Grok-4.1"
            client = OpenAI(
                api_key=os.getenv("POE_API_KEY"), base_url="https://api.poe.com/v1"
            )
        
        case "deepseek":
            model = "deepseek-chat"
            client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
            )
        case "gpt":
            model = "gpt-4o-mini"
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

    system_prompt = """
    你是孕婦健康及初生嬰兒育兒專家、嬰兒奶粉品牌專家, 永遠以JSON格式回應。
    用戶會提供嬰兒奶粉品牌名稱, 品牌相關關鍵字, 及奶粉品牌相關的 WhatsApp 媽媽群組對話給你, 請判斷用戶提供的文本對提及的品牌情緒是正面(P)、負面(N)、中立(I), 並提供情緒判斷的原因。必須使用以下JSON格式回應:
    
    {
        "sentiment": "sentiment code, either 'P', 'N' or 'I'",
        "reason": "reason of sentiment within 50 traditional Chinese characters"
    }
    
    Do not include any text outside the JSON object. Strictly adhere to this format.    
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
    ]

    messages.append({"role": "user", "content": user_prompt})
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, stream=False
        )
        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        return answer
    except APIStatusError as e:
        print(e)
        return 1

def main():
    pass
    
    if False:
        model = "deepseek-chat"
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        model = "gpt-4o-mini"
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    system_prompt = "你是孕婦健康及初生嬰兒育兒專家, 請判斷用戶提供的文本情緒是正面(positive)、負面(negative)、中立(indifferent), 永遠只回答相應的情緒編緒"
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    while True:
        user_input = input("Questions (input 'q' to quit): ").strip()
        if user_input.lower() == "q":
            break
        messages.append({"role": "user", "content": user_input})
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, stream=False
            )
            answer = response.choices[0].message.content
            messages.append({"role": "assistant", "content": answer})
            print(answer)
        except APIStatusError as e:
            print(e)


if __name__ == "__main__":
    main()

