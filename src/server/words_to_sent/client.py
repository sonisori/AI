import requests

server_url = 'http://127.0.0.1:5001/predict'

data = {'prediction': ['안녕하세요', '목', '아프다', '오다']}

# 데이터 전송
response = requests.post(server_url, json=data)

# 응답 출력
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
