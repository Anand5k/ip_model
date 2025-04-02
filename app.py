import os
import requests
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Google Drive Direct Download Link
MODEL_URL = "https://drive.google.com/uc?id=1CFLzcTN41YIWwFBTE9q0luEETHt_lduh"
MODEL_PATH = "image_classifier.keras"
class_labels = ["Algae", "Black Crust", "Crack", "Erosion", "Graffiti"]

# Function to download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✅ Model downloaded successfully.")
        else:
            raise Exception("❌ Failed to download model. Check the link.")

# Download and load the model
download_model()
model = load_model(MODEL_PATH)

# Function for enhanced preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    # 1️⃣ Gamma Correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # 2️⃣ Bilateral Filtering
    img_filtered = cv2.bilateralFilter(img_gamma, d=9, sigmaColor=75, sigmaSpace=75)

    # 3️⃣ Saturation Enhancement
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    img_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    img_array = img_final.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# API Endpoint to Predict Image Type
@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # Save uploaded image
        image_file = request.files['image']
        image_path = "uploaded_image.jpg"
        image_file.save(image_path)

        # Preprocess and Predict
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
