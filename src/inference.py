import onnxruntime as ort
import numpy as np
import json
from PIL import Image
import os


class PetClassifier:
    def __init__(
        self, model_path="results/model.onnx", labels_path="results/classes.json"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found at {labels_path}")

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        # Start ONNX session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path, sess_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_data = np.array(image).astype(np.float32)

        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).astype(np.float32)
        image_data = image_data / 255.0
        image_data = (image_data - mean) / std

        # Transpose to (C, H, W)
        image_data = image_data.transpose(2, 0, 1)

        # Add batch dimension
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def predict(self, image_path):
        input_data = self.preprocess(image_path)
        inputs = {self.input_name: input_data}
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]
        class_idx = np.argmax(logits)
        return self.classes[class_idx]


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pet class from image")
    parser.add_argument(
        "image_path", nargs="?", default="test_image.jpg", help="Path to the image file"
    )
    args = parser.parse_args()

    # Create a dummy image for testing if default is used and doesn't exist
    if args.image_path == "test_image.jpg" and not os.path.exists("test_image.jpg"):
        img = Image.new("RGB", (300, 300), color="red")
        img.save("test_image.jpg")
        print("Created dummy test_image.jpg")

    if not os.path.exists(args.image_path):
        print(f"Error: Image {args.image_path} not found.")
    else:
        classifier = PetClassifier()
        prediction = classifier.predict(args.image_path)
        print(f"Prediction for {args.image_path}: {prediction}")
