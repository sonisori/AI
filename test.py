import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
from get_a_hand_data import *

def preprocess_data(res):
    joint = np.zeros((21, 4))
    for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

    # Compute angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
    v = v2 - v1  # [20, 3]

    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # v의 길이로 나눠줌

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,] 15개의 각도 구함

    angle = np.degrees(angle)  # Convert radian to degree

    d = np.concatenate([joint.flatten(), angle])  # data concat
    return d

# 웹캠 정보
img_width = 1920
img_height = 1080

# 텍스트 화면에 출력시 사용할 정보
x_word = 50
y_word = 100
x_word_set = 50
y_word_set = img_height - 50
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 3
color = (0, 255, 0)
thickness = 4

actions = ['want','hello','reservation','hospital']
seq_length = 30

model = load_model('models/model.h5')

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

words_set = []

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
        for res in result.multi_hand_landmarks:
            d = preprocess_data(res)
            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # 결과 판단 위해 y_pred에 뽑아내기
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred)) # index 추출
            conf = y_pred[i_pred] # confidence 추출

            if conf < 0.9: # 90이하면 포즈 취하지 않은 걸로 생각
                continue

            # 90 넘으면
            action = actions[i_pred] # action 저장
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?' # 3번 반복되지 않으면 ? 출력
            # action 판단 로직: 마지막 action 3개가 전부 동일할 때 유효한 action이라고 판단 -> 오류 줄임
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action


            if this_action not in words_set and this_action != "?":  # 중복 체크
                words_set.append(this_action)

            # print(this_action)
            # print(words_set)
            print(img.shape)
            cv2.putText(img, "now: "+this_action, (x_word,y_word), fontFace, fontScale, color, thickness)


    cv2.putText(img, "now: ", (x_word,y_word), fontFace, fontScale, color, thickness)
    words_string = ", ".join(map(str, words_set))
    cv2.putText(img, words_string, (x_word_set,y_word_set), fontFace, fontScale, color, thickness)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break