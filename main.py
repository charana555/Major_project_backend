from flask import Flask, request, jsonify
from model import Prediction
import cv2  
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = "lung_cancer.h5"
predictor = Prediction(model_path)

@app.route("/predict", methods=["POST"])
def predict_image():
    # Check if image is present in request
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    image = request.files["image"]

    # Try-except block for error handling
    try:
        # Read image bytes into memory
        image_bytes = image.read()

        # Decode image bytes using OpenCV (replace with Pillow if needed)
        image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # Assuming color image

        # Pass the decoded image array for prediction
        response = predictor.predict(image_array)
        return response, 200
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
