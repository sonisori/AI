from flask import Flask, request, jsonify
from gpt import *

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # 데이터 받기
    prediction_list = data.get('prediction', [])
    prediction_sentence = runGPT(prediction_list)
    response = {'prediction_sentence': prediction_sentence}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # 서버 실행
