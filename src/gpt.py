from openai import OpenAI
from dotenv import load_dotenv
import os

def runGPT(words_list_korean, senType):
    if senType == 0:
        sentence_type = "평서문"
    elif senType == 1:
        sentence_type = "의문문"
    elif senType == 2:
        sentence_type = "감탄문"

    load_dotenv()

    gpt_key = os.getenv('GPT_KEY')
    client = OpenAI(api_key= gpt_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "system", "content": "당신은 최고의 한국 수어 통역사 입니다."},
        {"role": "user", "content": "농인이 동작한 몇 개의 수어 단어들을 한국어 문장으로 해석하려고 합니다."},
        {"role": "user", "content": "다음 순서가 고려된 수어 단어 리스트가 있습니다."},
        {"role": "user", "content": "자연스럽고 완전한 한국어 존댓말 문장을 생성하세요."},
        {"role": "user", "content": "불필요한 단어는 추가하지 마세요."},
        {"role": "user", "content": f"{senType}으로 작성하세요."},
        {"role": "user", "content": f"{words_list_korean}"},
        ],
        temperature=0.5
    )
    result = response.choices[0].message.content
    return result

