from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

MODEL_FILENAME = "knn_model.joblib"
SCALER_FILENAME = "scaler.joblib"

knn_model = joblib.load(MODEL_FILENAME)
scaler = joblib.load(SCALER_FILENAME)

label_map = {
    0: "Good (Baik)",
    1: "Moderate (Sedang)",
    2: "Unhealthy (Buruk)"
}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


    

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_keys = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Check input
    for key in required_keys:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"})

    features = [float(data[k]) for k in required_keys]
    arr = np.array([features])
    arr_scaled = scaler.transform(arr)

    pred_class = int(knn_model.predict(arr_scaled)[0])
    pred_label = label_map[pred_class]

    return jsonify({
        "predicted_class": pred_class,
        "predicted_label": pred_label
    })

if __name__ == "__main__":
    app.run(debug=True)
