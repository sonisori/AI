###### test 1: predict
import socketio
import requests

# Socket.IO 클라이언트 생성
sio = socketio.Client()

result = {
    "landmarks": [[{'x': 0.5927601456642151, 'y': 0.9861617088317871, 'z': 5.241136591394024e-07}, {'x': 0.5155649781227112, 'y': 0.9443425536155701, 'z': -0.02644294686615467}, {'x': 0.4653223156929016, 'y': 0.8654327988624573, 'z': -0.05608806386590004}, {'x': 0.4283157289028168, 'y': 0.8055499792098999, 'z': -0.08550803363323212}, {'x': 0.3879893720149994, 'y': 0.7522481679916382, 'z': -0.11727960407733917}, {'x': 0.5201188921928406, 'y': 0.7134290337562561, 'z': -0.0607912577688694}, {'x': 0.4643537402153015, 'y': 0.6126284599304199, 'z': -0.11317681521177292}, {'x': 0.40943068265914917, 'y': 0.5712025165557861, 'z': -0.1540045291185379}, {'x': 0.35416293144226074, 'y': 0.5523128509521484, 'z': -0.1807226538658142}, {'x': 0.5629892349243164, 'y': 0.7103217840194702, 'z': -0.08040456473827362}, {'x': 0.5413597822189331, 'y': 0.5705279111862183, 'z': -0.1297880858182907}, {'x': 0.4860770106315613, 'y': 0.5005118250846863, 'z': -0.166713148355484}, {'x': 0.42599019408226013, 'y': 0.4596896767616272, 'z': -0.19132985174655914}, {'x': 0.6062142848968506, 'y': 0.7376664876937866, 'z': -0.10296988487243652}, {'x': 0.6074603199958801, 'y': 0.5974946022033691, 'z': -0.14850078523159027}, {'x': 0.5631113648414612, 'y': 0.5269138813018799, 'z': -0.1770201474428177}, {'x': 0.5126218795776367, 'y': 0.4768882095813751, 'z': -0.19573284685611725}, {'x': 0.6467544436454773, 'y': 0.790822446346283, 'z': -0.1265859156847}, {'x': 0.6644154787063599, 'y': 0.6943660974502563, 'z': -0.16781388223171234}, {'x': 0.6445795297622681, 'y': 0.6323617100715637, 'z': -0.18642652034759521}, {'x': 0.6124258041381836, 'y': 0.5804388523101807, 'z': -0.198034405708313}],
                [{'x': 0.19248434901237488, 'y': 1.1097540855407715, 'z': 6.681350441795075e-07}, {'x': 0.2783002257347107, 'y': 1.0505576133728027, 'z': -0.03449355438351631}, {'x': 0.3402406573295593, 'y': 0.9515190720558167, 'z': -0.05266229063272476}, {'x': 0.3675280511379242, 'y': 0.858670711517334, 'z': -0.06854456663131714}, {'x': 0.38749271631240845, 'y': 0.7678257822990417, 'z': -0.08708208799362183}, {'x': 0.2668186128139496, 'y': 0.8393048048019409, 'z': -0.05674349516630173}, {'x': 0.2830912470817566, 'y': 0.7137707471847534, 'z': -0.09529367089271545}, {'x': 0.30522269010543823, 'y': 0.6548000574111938, 'z': -0.12243613600730896}, {'x': 0.3318330645561218, 'y': 0.6092461347579956, 'z': -0.1423996239900589}, {'x': 0.20551344752311707, 'y': 0.8441810607910156, 'z': -0.06613518297672272}, {'x': 0.19969668984413147, 'y': 0.7094354033470154, 'z': -0.10935661196708679}, {'x': 0.21733810007572174, 'y': 0.6331965923309326, 'z': -0.14348138868808746}, {'x': 0.2427501529455185, 'y': 0.5719804763793945, 'z': -0.16865700483322144}, {'x': 0.14677613973617554, 'y': 0.8773866891860962, 'z': -0.0788705050945282}, {'x': 0.13092710077762604, 'y': 0.7560811638832092, 'z': -0.12577977776527405}, {'x': 0.1328216791152954, 'y': 0.6705409288406372, 'z': -0.15626588463783264}, {'x': 0.1460469663143158, 'y': 0.5936620235443115, 'z': -0.1756393015384674}, {'x': 0.09486538171768188, 'y': 0.9295968413352966, 'z': -0.09288428723812103}, {'x': 0.07266755402088165, 'y': 0.8357580304145813, 'z': -0.13758084177970886}, {'x': 0.06587590277194977, 'y': 0.7697023749351501, 'z': -0.16010122001171112}, {'x': 0.0674748569726944, 'y': 0.7033976912498474, 'z': -0.17436087131500244}]],
    "time": "12345"
}

landmarks = result["landmarks"]

# 서버 연결 시 호출
@sio.event
def connect():
    print("서버에 연결되었습니다.")

# 'prediction_result' 이벤트로 예측 결과 수신
@sio.on('prediction_result')
def on_prediction_result(data):
    print("예측 결과:", data)

# 에러 발생 시 에러 수신
@sio.on('error')
def on_error(data):
    print("에러:", data)

# 서버 연결 해제 시 호출
@sio.event
def disconnect():
    print("서버와 연결이 끊어졌습니다.")

# 'predict' 이벤트로 테스트 데이터 전송
def test_predict_event():
    print("테스트 데이터 전송:", landmarks)
    while(1):
        c = input()
        sio.emit('predict', landmarks)  # 서버에 predict 이벤트 전송
        if(c=='.'):
            break

# 서버 연결 및 테스트 실행
sio.connect('http://localhost:5002')  # 서버 주소와 포트 설정
test_predict_event()
sio.wait()  # 서버의 응답을 계속 기다림


# ###### test 2: gpt_make_sentence
# import requests
# import json
# #
# # # 서버의 URL (서버가 실행 중이어야 합니다.)
# url = 'http://localhost:5002/makeSentence0'  # 서버의 URL을 실제 URL로 바꿔주세요
#
# # 테스트 데이터 예시
# data = {
#     "prediction": ["안녕하세요", "목", "아프다", "오다"]
# }
#
# # POST 요청 보내기
# response = requests.post(url, json=data)
#
# # 응답 결과 출력
# if response.status_code == 200:
#     result = response.json()  # 응답 데이터 JSON 파싱
#     print("응답 결과:", result)
# else:
#     print(f"요청 실패. 상태 코드: {response.status_code}")
#
# ####### test 3: gpt_evaluate_meaning
#
# import requests
# import json
#
# # 서버의 URL (서버가 실행 중이어야 합니다.)
# url = 'http://localhost:5002/evaluateMeaning'  # 서버의 URL을 실제 URL로 바꿔주세요
#
# # 테스트 데이터 예시
# data = {
#     "prediction": ["주택", "담보", "대출", "알아보다", "오다"],  # 예시 단어 리스트
#     "quiz_index": 12  # 퀴즈 ID (데이터베이스나 파일에서 퀴즈 정보를 가져오는 인덱스 값)
# }
#
# # POST 요청 보내기
# response = requests.post(url, json=data)
#
# # 응답 결과 출력
# if response.status_code == 200:
#     result = response.json()  # 응답 데이터 JSON 파싱
#     print("응답 결과:", result)
# else:
#     print(f"요청 실패. 상태 코드: {response.status_code}")
#
