import os
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

def sentiment_check(user_prompt: str):
    load_dotenv()
    if True:
        model = "deepseek-chat"
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        model = "gpt-4o-mini"
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    system_prompt = "你是孕婦健康及初生嬰兒育兒專家, 用戶會提供嬰兒奶粉品牌名稱, 品牌相關關鍵字, 及奶粉品牌相關的媽媽WhatsApp群組對話給你, 請判斷用戶提供的文本對提及的品牌情緒是正面(P)、負面(N)、中立(I), 永遠只回答相應的情緒代號"
    messages = [
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

