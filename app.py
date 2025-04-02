import os
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1XIRebSrOIXU7n08sXOfCrIQbmyiRIELW"
MODEL_PATH = "/tmp/image_classifier.keras"
CLASS_LABELS = ["Algae", "Black Crust", "Crack", "Erosion", "Graffiti"]

# Download model if not available
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully")

# Load the model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Preprocessing function
def preprocess_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    # 1️⃣ Gamma Correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # 2️⃣ Bilateral Filtering
    img_filtered = cv2.bilateralFilter(img_gamma, d=9, sigmaColor=75, sigmaSpace=75)

    # 3️⃣ Increase Saturation
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    img_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Normalization
    img_array = img_final.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_file = request.files['image']
        input_data = preprocess_image(image_file)
        predictions = model.predict(input_data)
        class_name = CLASS_LABELS[np.argmax(predictions)]
        confidence = np.max(predictions)

        return jsonify({
            "prediction": class_name,
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check route
@app.route("/", methods=["GET"])
def health_check():
    return "Server is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
