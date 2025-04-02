import numpy as np
import os
import cv2
import gdown
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Google Drive link for the model
DRIVE_URL = "YOUR_NEW_DRIVE_LINK_WITH_H5"
MODEL_PATH = "image_classifier.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model download failed!")

# Load the trained model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Define class labels
class_labels = ["Algae", "Black Crust", "Crack", "Erosion", "Graffiti"]

# Function for enhanced preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (150, 150))  # Resize to match model input

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

    img_array = img_final.astype(np.float32) / 255.0  # Normalize (0-1 scale)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = "uploaded_image.jpg"
    image_file.save(image_path)

    try:
        # Preprocess image and predict
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(image_path)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
