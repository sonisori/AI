from openai import OpenAI
# from dotenv import load_dotenv
import os

def gpt_make_sentence(words_list, senType):
    if senType == 0:
        sentence_type = "평서문"
    elif senType == 1:
        sentence_type = "의문문"
    elif senType == 2:
        sentence_type = "감탄문"

    # load_dotenv()

    gpt_key = os.environ.get('GPT_KEY')
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
        {"role": "user", "content": f"{words_list}"},
        ],
        temperature=0.5
    )
    result = response.choices[0].message.content
    return result

def gpt_evaluate_meaning(sentence, words_list):
    # load_dotenv()


    gpt_key = os.environ.get('GPT_KEY')
    client = OpenAI(api_key= gpt_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "system", "content": "당신은 언어 전문가입니다."},
        {"role": "user", "content": "지금부터 주어질 단어 리스트와 정답 문장을 비교하여, 두 문장이 의미적으로 동일한지 예/아니오로 답변해 주세요."},
        {"role": "user", "content": f"- 정답 문장: {sentence}"},
        {"role": "user", "content": f"- 단어 리스트: {words_list}"},
        ],
        temperature=0.5
    )
    result = response.choices[0].message.content
    return result