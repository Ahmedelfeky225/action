from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np

app = Flask(__name__)


CORS(app)

api = Api(app)

model = joblib.load(open('nM.pkl', 'rb'))


app.route('/')
def home():
    return 'Corona result api 😧 '


@app.route("/predict", methods=["post"])
def predict():
    result = request.json
    quary_df = pd.DataFrame(result)
    prediction = model.predict(quary_df)
    return jsonify({"prediction": list(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
