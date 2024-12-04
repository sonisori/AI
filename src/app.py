from flask import Flask, session, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
import mysql.connector
from gpt import *

seq_length = 30

model = tf.keras.models.load_model('./models/model.keras')

def preprocess_data_server(res):
    joint = np.array([[lm['x'], lm['y'], lm['z']] for lm in res])

    # Compute angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]]  # Parent joint
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]  # Child joint
    v = v2 - v1  # [20, 3]

    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # v의 길이로 나눠줌

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,] 15개의 각도 구함

    angle = np.degrees(angle)  # Convert radian to degree

    d = np.concatenate([joint.flatten()*500, angle])  # data concat

    return d

def get_words_by_ids(id_list):
    connection = mysql.connector.connect(
        host='your_host',  # 예: 'localhost' 또는 '127.0.0.1'
        user='your_user',  # 사용자 이름
        password='your_password',  # 비밀번호
        database='sonisori'  # 데이터베이스 이름
    )

    cursor = connection.cursor()

    id_tuple = tuple(id_list)
    query = f"SELECT id, word FROM sign_words WHERE id IN ({','.join(['%s'] * len(id_tuple))})"
    cursor.execute(query, id_tuple)
    result = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.close()
    connection.close()

    return [result.get(id_value, None) for id_value in id_list]

# Flask 및 SocketIO 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app,cors_allowed_origins="*")

@app.route('/')
def hello_world():
    return ''

@socketio.on('connect')
def create_session():
    session['seq'] = []
    session['action_seq'] = []
    session['id_list'] = []
    session['word_list'] = []

# 랜드마크 데이터 seq -> 수어 Id
@socketio.on('predict')
def handle_predict(data):
    seq = session['seq']
    action_seq = session['action_seq']
    id_list = session['id_list']
    word_list = session['word_list']

    try:
        for d in data:
            d = preprocess_data_server(d)
            seq.append(d)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.8:
                continue

            action = i_pred
            action_seq.append(action)

            if len(action_seq) < 15:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[-6] == action_seq[-7] == action_seq[-8] == action_seq[-9] == action_seq[-10] == action_seq[-11] == action_seq[-12] == action_seq[-13] == action_seq[-14] == action_seq[-15]:
                this_action = action

            if this_action not in id_list and this_action != "?":  # 중복 체크
                id_list.append(this_action)
                # word_list = get_words_by_ids(id_list)
                # 예측 결과를 리스트로 변환 후 클라이언트에게 전송
                emit('prediction_result', {'prediction': id_list,'appended':this_action}) # $$$
                # emit('prediction_result', {'prediction': word_list,'appended':this_action}) # ***
        print("result: ", id_list) # $$$
        # print("result: ", word_list) # ***

    except Exception as e:
        # 에러 발생 시 에러 메시지를 클라이언트에 전송
        emit('error', {'error': str(e)})

# ['안녕하세요', '목', '아프다', '오다'] -> '안녕하세요, 목이 아파서 왔습니다.'
@app.route('/makeSentence', methods=['POST'])
def predict():
    data = request.get_json() # 데이터 받기
    prediction_list = data.get('prediction', [])
    prediction_sentence = runGPT(prediction_list)
    response = {'prediction_sentence': prediction_sentence}
    return jsonify(response)

# 서버 실행
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)