from flask import Flask, session
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
from preprocess_data import *

actions = ['hello','bye']
seq_length = 30

model = tf.keras.models.load_model('models/model.keras')



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
    session['words_set'] = []

# 클라이언트가 'predict' 이벤트로 데이터를 보낼 때 실행
@socketio.on('predict')
def handle_predict(data):
    seq = session['seq']
    action_seq = session['action_seq']
    words_set = session['words_set']
    
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

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            if this_action not in words_set and this_action != "?":  # 중복 체크
                words_set.append(this_action)
                # 예측 결과를 리스트로 변환 후 클라이언트에게 전송
                emit('prediction_result', {'prediction': words_set,'appended':this_action})
        print("result: ",words_set)

    except Exception as e:
        # 에러 발생 시 에러 메시지를 클라이언트에 전송
        emit('error', {'error': str(e)})

# 서버 실행
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)