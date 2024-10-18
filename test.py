import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['love','thanks']
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

while cap.isOpened(): # 카메라 열려 있는 동안
    ret, img = cap.read() # 한 프레임씩 읽기
    img = img.copy()

    # openCV: BGR, mediapipe: RGB
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img) # mediapipe에 넣기 전 이미지 전처리
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지 출력을 위해 다시 바꿔주기

    if result.multi_hand_landmarks is not None: # 손을 인식했으면
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # v의 길이로 나눠줌

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,] 15개의 각도 구함

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle]) # data concat

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
            org = (50, 100)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 4
            color = (255, 255, 255)
            thickness = 3

            cv2.putText(img, f'{this_action.upper()}', org, fontFace, fontScale, color, thickness)
            # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break