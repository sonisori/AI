from flask import Flask, session, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
import pymysql
# from dotenv import load_dotenv
import os
from gpt import *
from flask_cors import CORS

# load_dotenv()

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

def get_word_by_id(index):
    db_config = {
        "host": os.environ.get("DB_HOST"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "database": os.environ.get("DB_NAME"),
    }
    connection = pymysql.connect(**db_config)
    index = index+1
    try:
        cursor = connection.cursor()

        query = "SELECT word FROM sign_words WHERE id = %s;"
        cursor.execute(query, (index,))

        result = cursor.fetchone()

        return result[0] if result else None

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return None

    finally:
        if connection:
            connection.close()
# print(get_word_by_id(1))
def get_quiz_by_id(index):
    db_config = {
        "host": os.environ.get("DB_HOST"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "database": os.environ.get("DB_NAME"),
    }
    connection = pymysql.connect(**db_config)
    try:
        cursor = connection.cursor()

        query = "SELECT sentence FROM sign_quizzes WHERE id = %s;"
        cursor.execute(query, (index,))

        result = cursor.fetchone()

        return result[0] if result else None

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        return None

    finally:
        if connection:
            connection.close()

# Flask 및 SocketIO 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app,cors_allowed_origins="*")

CORS(app)

@app.route('/')
def home():
    return "<h1>AI Server</h1>"

@socketio.on('connect')
def create_session():
    session['seq'] = []
    session['action_seq'] = []
    session['id_list'] = []
    session['word_list'] = []

last_time_ms = 0

# 랜드마크 데이터 seq -> 수어 Id
@socketio.on('predict')
def handle_predict(data):
    seq = session['seq']
    action_seq = session['action_seq']
    id_list = session['id_list']
    word_list = session['word_list']

    print("data:",data)

    global last_time_ms
    current_time_ms = data["time"]

    if current_time_ms - last_time_ms > 100:
        landmarks = data["landmarks"]

        try:
            for d in landmarks:
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

                if len(action_seq) < 10:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[
                    -6] == action_seq[-7] == action_seq[-8]:
                    this_action = action

                if this_action not in id_list and this_action != "?":  # 중복 체크
                    id_list.append(this_action)
                    this_word = get_word_by_id(this_action)
                    word_list.append(this_word)
                    # 예측 결과를 리스트로 변환 후 클라이언트에게 전송
                    # emit('prediction_result', {'prediction': id_list,'appended':this_action}) # $$$
                    emit('prediction_result', {'prediction': word_list, 'appended': this_action})  # ***
            # print("result: ", id_list) # $$$
            print("result: ", word_list)  # ***

        except Exception as e:
            # 에러 발생 시 에러 메시지를 클라이언트에 전송
            emit('error', {'error': str(e)})

        last_time_ms = current_time_ms


# 단어리스트 -> 문장
# e.g. ['안녕하세요', '목', '아프다', '오다'] -> '안녕하세요, 목이 아파서 왔습니다.'
# 0: 평서문
@app.route('/makeSentence0', methods=['POST'])
def make_sentence0():
    data = request.get_json() # 데이터 받기
    prediction_list = data.get('prediction', [])
    prediction_sentence = gpt_make_sentence(prediction_list,0)
    response = {'prediction_sentence': prediction_sentence}
    return jsonify(response)

# 1: 의문문
@app.route('/makeSentence1', methods=['POST'])
def make_sentence1():
    data = request.get_json() # 데이터 받기
    prediction_list = data.get('prediction', [])
    prediction_sentence = gpt_make_sentence(prediction_list,1)
    response = {'prediction_sentence': prediction_sentence}
    return jsonify(response)

# 2: 감탄문
@app.route('/makeSentence2', methods=['POST'])
def make_sentence2():
    data = request.get_json() # 데이터 받기
    prediction_list = data.get('prediction', [])
    prediction_sentence = gpt_make_sentence(prediction_list,2)
    response = {'prediction_sentence': prediction_sentence}
    return jsonify(response)

# 문장과 단어리스트가 의미적으로 동일한지 (예(1)/아니오(0))로 판단
# e.g. "주택 담보 대출을 알아보려고 왔어요.", ["주택", "담보", "대출", "알아보다", "오다"] -> 예
@app.route('/evaluateMeaning', methods=['POST'])
def evaluate_meaning():
    data = request.get_json() # 데이터 받기
    prediction_list = data.get('prediction', [])
    index = data.get('quiz_index', [])

    quiz_sentence = get_quiz_by_id(index)

    result = gpt_evaluate_meaning(quiz_sentence, prediction_list)
    response = {'result': result}
    if result=="예":
        return jsonify(response)
    elif result=="아니오":
        return jsonify(response)
    else:
        return jsonify({'error': 'Results other than Yes/No were output.'}), 400

# 서버 실행
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, debug=True, allow_unsafe_werkzeug=True)