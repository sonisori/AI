from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# @app.route('/')
# def hello_world():
#     return 'Hello World!'

# 모델 로드
model = tf.keras.models.load_model('models/model.h5')

# 데이터 전처리 함수
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

# 예측
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        processed_data = preprocess_data(data)
        prediction = model.predict(processed_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 서버 상태 체크
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True) # debug=True로 수정 사항 실시간으로 반영
