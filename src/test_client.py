import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
from preprocess_data import *
import socketio
import requests
import re
import json

# Socket.IO 클라이언트 생성
sio = socketio.Client()
@sio.event
def connect():
    print("서버에 연결되었습니다.")
@sio.event
def disconnect():
    print("서버와 연결이 끊어졌습니다.")
# 웹캠 정보
img_width = 1920
img_height = 1080

model = load_model('models/model.keras')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)
# input_video_path = './db_video/action1.mov'
# cap = cv2.VideoCapture(input_video_path)

seq = []
action_seq = []

frame_count = 0
start_time = time.time()

words_list = []


def convert_landmarks_to_json(data, time_ms):
    x = np.zeros(21)
    y = np.zeros(21)
    z = np.zeros(21)

    for j, lm in enumerate(data.landmark):
        x[j]=lm.x
        y[j]=lm.y
        z[j]=lm.z

    landmarks = [{"x": float(x[i]), "y": float(y[i]), "z": float(z[i])} for i in range(21)]
    result = {"landmarks": [landmarks],
              "time": time_ms
              }

    return result

def test_predict_event():

    while cap.isOpened(): # 카메라 열려 있는 동안
        ret, img = cap.read() # 한 프레임씩 읽기
        img = img.copy()

        # openCV: BGR, mediapipe: RGB
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img) # mediapipe에 넣기 전 이미지 전처리
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지 출력을 위해 다시 바꿔주기

        if result.multi_hand_landmarks is not None: # 손을 인식했으면
            current_time_ms = int(time.time() * 1000)
            # print("result: ", result.multi_hand_landmarks)
            # print("result: ", type(result))
            for res in result.multi_hand_landmarks:
                # print("res: ", res)
                data = convert_landmarks_to_json(res,current_time_ms)
                print(data)
                # here!! #
                sio.emit('predict', data)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


sio.connect('http://localhost:5002')  # 서버 주소와 포트 설정
test_predict_event()
sio.wait()  # 서버의 응답을 계속 기다림