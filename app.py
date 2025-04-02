import os
import zipfile
import requests
import numpy as np
import cv2
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Google Drive Link for ZIP file
ZIP_URL = "https://drive.google.com/uc?id=1-3UFKJrMrddq-Oo0sfKfHe2nMMI_ghOt"
MODEL_DIR = "model_dir"
MODEL_PATH = os.path.join(MODEL_DIR, "image_classifier.h5")

# ✅ Step 1: Download and Extract Model
def download_and_extract_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already extracted!")
        return

    print("🔄 Downloading and extracting model...")
    response = requests.get(ZIP_URL)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("✅ Model extracted successfully!")

# ✅ Step 2: Load the Model
try:
    download_and_extract_model()
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# ✅ Step 3: Define Class Labels
class_labels = ["Algae", "Black Crust", "Crack", "Erosion", "Graffiti"]

# ✅ Step 4: Preprocessing Function
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (150, 150))           # Resize for model input

    # ⚡ 1. Gamma Correction to Prevent Over-Enhancement
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # ⚡ 2. Bilateral Filtering for Noise Reduction
    img_filtered = cv2.bilateralFilter(img_gamma, d=9, sigmaColor=75, sigmaSpace=75)

    # ⚡ 3. Increase Saturation for Better Classification
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    img_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Normalize
    img_array = img_final.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ✅ Step 5: Prediction Function
def classify_image(img_path):
    try:
        # Preprocess the Image
        img_array = preprocess_image(img_path)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "details": {label: float(score) for label, score in zip(class_labels, prediction[0])}
        }
    except Exception as e:
        return {"error": str(e)}

# ✅ Step 6: Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_path = os.path.join(MODEL_DIR, file.filename)
    file.save(img_path)

    result = classify_image(img_path)
    return jsonify(result)

# ✅ Step 7: Run the Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
