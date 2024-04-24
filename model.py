import cv2
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import load_model  # Assuming TensorFlow backend

class Prediction:
    def __init__(self, model_path):
        # Ensure model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = load_model(model_path)  # Load the .h5 model

    def preprocess_image(self, image):
        # Assuming image is a NumPy array (BGR format)
        resized_image = cv2.resize(image, (256, 256))
        return resized_image

    def predict(self, image):
        preprocessed_image = self.preprocess_image(image)
        prediction = self.model.predict(np.expand_dims(preprocessed_image, axis=0))[0]  # Add batch dimension

        # Convert prediction to a list (if needed)
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()

        percentages = [p * 100 for p in prediction]   

        return json.dumps({"prediction": percentages})
