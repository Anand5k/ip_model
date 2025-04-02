import os
import requests
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?id=1A_-Iu_mmD7xGLtF_WIAXPpGbdLKenVv5"
MODEL_PATH = "image_classifier.h5"

# Class labels
class_labels = ["Algae", "Black Crust", "Crack", "Erosion", "Graffiti"]

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already exists.")

# Download and load the model
download_model()
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Preprocessing function
def preprocess_image(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    # Gamma Correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # Bilateral Filtering
    img_filtered = cv2.bilateralFilter(img_gamma, d=9, sigmaColor=75, sigmaSpace=75)

    # Saturation Enhancement
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    img_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    img_array = img_final.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    try:
        img_array = preprocess_image(image_data)
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
