import os
from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import pickle

app = Flask(__name__)
cors=CORS(app,resources={r'/*':{'origins':'*'}})
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('model_regr.pkl','rb'))

@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def main():
    if request.method == 'POST':
        data = request.get_json()
        int_features = [float(x) for x in data.values()]
        final_features = [np.array(int_features)]
        prediction = (model.predict(final_features)).tolist()
        predict_approx = str(round(prediction[0], 2))
        return predict_approx

    if request.method == 'GET':
        return "Flask API is running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)