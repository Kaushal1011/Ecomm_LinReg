from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


def get_model(path: str = "./model.pickle") -> LinearRegression:
    return pickle.load(open(path, "rb"))


app = Flask(__name__)
model = get_model()


@app.route("/")
def homepage():
    addr = request.remote_addr

    curl = """curl --location --request POST '{}/predict' \
--header 'Content-Type: application/json' \
--data-raw '{{
"time_on_app":12.655651,
"Length_of_membership":4.082621,
"time_on_website":39.577668,
"avg_session_length":34.497268
}}'""".format(
        addr
    )

    return curl


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run()
