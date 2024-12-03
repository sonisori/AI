import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
from preprocess_data import *

# 웹캠 정보
img_width = 1920
img_height = 1080

# 텍스트 화면에 출력시 사용할 정보
x_word = 50

y_word = 100
x_word_list = 50
y_word_list = img_height - 50
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 3
color = (0, 255, 0)
thickness = 4

seq_length = 30

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

while cap.isOpened(): # 카메라 열려 있는 동안
    ret, img = cap.read() # 한 프레임씩 읽기
    img = img.copy()

    frame_count += 1
    # 1초가 지나면 FPS 계산
    if time.time() - start_time >= 1.0:
        print(f'FPS: {frame_count}')
        frame_count = 0
        start_time = time.time()

    # openCV: BGR, mediapipe: RGB
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img) # mediapipe에 넣기 전 이미지 전처리
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지 출력을 위해 다시 바꿔주기

    if result.multi_hand_landmarks is not None: # 손을 인식했으면
        # print("result: ", result.multi_hand_landmarks)
        # print("result: ", type(result))
        for res in result.multi_hand_landmarks:
            # print("res: ", res)
            d = preprocess_data_test(res)
            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            if conf < 0.8:
                continue
            print(i_pred, conf)

            action = i_pred # action 저장
            action_seq.append(action)

            if len(action_seq) < 15:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[-6] == action_seq[-7] == action_seq[-8] == action_seq[-9] == action_seq[-10] == action_seq[-11] == action_seq[-12] == action_seq[-13] == action_seq[-14] == action_seq[-15]:
                this_action = action

            if this_action not in words_list and this_action != "?":  # 중복 체크
                words_list.append(this_action)

            cv2.putText(img, "now: "+ str(this_action), (x_word,y_word), fontFace, fontScale, color, thickness)


    cv2.putText(img, "now: ", (x_word,y_word), fontFace, fontScale, color, thickness)
    words_string = ", ".join(map(str, words_list))
    cv2.putText(img, words_string, (x_word_list,y_word_list), fontFace, fontScale, color, thickness)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break