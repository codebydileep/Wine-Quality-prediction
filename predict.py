from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("my_custom_model.pkl")
scaler = joblib.load("my_custom_scaler.pkl")

# Define feature names (Ensure they match training data)
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


@app.route("/")
def home():
    return render_template("index.html")  # Load frontend


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        input_values = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
