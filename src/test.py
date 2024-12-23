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

seq = []
action_seq = []

frame_count = 0
start_time = time.time()

words_list = []

previous_timestamp = None  # 이전 프레임의 timestamp 저장 변수

# 100ms 단위로 처리
frame_interval = 95  # ms
last_frame_time = 0  # 마지막으로 처리한 프레임의 timestamp(ms)

while cap.isOpened():  # 카메라 열려 있는 동안
    ret, img = cap.read()  # 한 프레임씩 읽기
    if not ret:
        break
    img = img.copy()

    # 현재 프레임의 timestamp(ms) 계산
    current_timestamp = int(time.time() * 1000)

    # 100ms 간격을 확인
    # if current_timestamp - last_frame_time < frame_interval:
    #     continue  # 100ms가 경과하지 않았다면 다음 프레임으로 넘어감

    last_frame_time = current_timestamp  # 마지막 처리 시간 갱신

    if previous_timestamp is not None:
        timestamp_diff = current_timestamp - previous_timestamp
        print(f'Timestamp (ms): {current_timestamp}, Diff (ms): {timestamp_diff}')
    else:
        print(f'Timestamp (ms): {current_timestamp}, Diff (ms): N/A (First frame)')

    # 현재 timestamp를 이전 timestamp로 업데이트
    previous_timestamp = current_timestamp

    frame_count += 1
    # 1초가 지나면 FPS 계산
    if time.time() - start_time >= 1.0:
        print(f'FPS: {frame_count}')
        frame_count = 0
        start_time = time.time()

    # openCV: BGR, mediapipe: RGB
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)  # mediapipe에 넣기 전 이미지 전처리
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 이미지 출력을 위해 다시 바꿔주기

    if result.multi_hand_landmarks is not None:  # 손을 인식했으면
        for res in result.multi_hand_landmarks:
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

            action = i_pred  # action 저장
            action_seq.append(action)

            if len(action_seq) < 11:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[-6] == action_seq[-7] == action_seq[-8] == action_seq[-9] == action_seq[-10] == action_seq[-11]:
                this_action = action

            if this_action not in words_list and this_action != "?":  # 중복 체크
                words_list.append(this_action)

            cv2.putText(img, "now: " + str(this_action), (x_word, y_word), fontFace, fontScale, color, thickness)

    cv2.putText(img, "now: ", (x_word, y_word), fontFace, fontScale, color, thickness)
    words_string = ", ".join(map(str, words_list))
    cv2.putText(img, words_string, (x_word_list, y_word_list), fontFace, fontScale, color, thickness)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

