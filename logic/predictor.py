"""
Machine Learning logic for image classification and preprocessing.
"""

import io
import random
from typing import Tuple
import numpy as np
from PIL import Image


import os
import json
import onnxruntime as ort


class ImagePredictor:
    """Class for image prediction and preprocessing."""

    def __init__(
        self, model_path="results/model.onnx", labels_path="results/classes.json"
    ):
        """
        Initialize the predictor with model and class names.
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.session = None
        self.classes = []

        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            try:
                # Load labels
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)

                # Start ONNX session
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(
                    self.model_path, sess_options, providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                print(f"Model loaded from {self.model_path}")
            # pylint: disable=broad-exception-caught
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(
                f"Model or labels not found at {self.model_path}, {self.labels_path}. Using fallback."
            )
            self.classes = ["cat", "dog"]  # Fallback

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        image_data = np.array(img).astype(np.float32)

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

    def predict(
        self, image_bytes: bytes = None, image_path: str = None, seed: int = None
    ) -> dict:
        """
        Predict the class of an image using ONNX model.
        Accepts either image_bytes or image_path.
        """
        if self.session is None:
            # Fallback to random if model not loaded
            if seed is not None:
                random.seed(seed)
            return {
                "predicted_class": random.choice(self.classes),
                "confidence": 0.0,
                "all_classes": self.classes,
            }

        try:
            if image_bytes is None:
                if image_path:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                else:
                    raise ValueError(
                        "Either image_bytes or image_path must be provided"
                    )

            input_data = self.preprocess(image_bytes)
            inputs = {self.input_name: input_data}
            outputs = self.session.run(None, inputs)
            logits = outputs[0][0]

            # Softmax for confidence
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            class_idx = np.argmax(probs)
            confidence = float(probs[class_idx])
            predicted_class = self.classes[class_idx]

            return {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "all_classes": self.classes,
            }
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "predicted_class": "error",
                "confidence": 0.0,
                "all_classes": self.classes,
            }

    def resize_image(
        self, image_path: str, width: int, height: int, output_path: str = None
    ) -> Tuple[int, int]:
        """
        Resize an image to specified dimensions.

        Args:
            image_path: Path to input image
            width: Target width
            height: Target height
            output_path: Path to save resized image (optional)

        Returns:
            Tuple of (new_width, new_height)
        """
        with Image.open(image_path) as img:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)

            if output_path:
                resized_img.save(output_path)

            return resized_img.size

    def resize_image_from_bytes(
        self, image_bytes: bytes, width: int, height: int, image_format: str
    ) -> bytes:
        """
        Resize an image from bytes.

        Args:
            image_bytes: Image data as bytes
            width: Target width
            height: Target height
            image_format: The format of the image (e.g., 'jpeg', 'png')

        Returns:
            Resized image as bytes
        """
        img = Image.open(io.BytesIO(image_bytes))
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)

        output_bytes = io.BytesIO()
        resized_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()

    def convert_to_grayscale(self, image_path: str, output_path: str = None) -> str:
        """
        Convert an image to grayscale.

        Args:
            image_path: Path to input image
            output_path: Path to save grayscale image (optional)

        Returns:
            Mode of the converted image
        """
        with Image.open(image_path) as img:
            grayscale_img = img.convert("L")

            if output_path:
                grayscale_img.save(output_path)

            return grayscale_img.mode

    def convert_to_grayscale_from_bytes(
        self, image_bytes: bytes, image_format: str
    ) -> bytes:
        """
        Convert an image from bytes to grayscale.

        Args:
            image_bytes: Image data as bytes
            image_format: The format of the image (e.g., 'jpeg', 'png')

        Returns:
            Grayscale image as bytes
        """
        img = Image.open(io.BytesIO(image_bytes))
        grayscale_img = img.convert("L")
        output_bytes = io.BytesIO()
        grayscale_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()

    def normalize_image(self, image_path: str, output_path: str = None) -> dict:
        """
        Normalize an image and save it.

        Args:
            image_path: Path to input image
            output_path: Path to save normalized image (optional)

        Returns:
            Dictionary with image statistics (mean, std)
        """
        with Image.open(image_path) as img:
            image_format = img.format
            img_array = np.array(img).astype(np.float32)

            mean = np.mean(img_array, axis=(0, 1))
            std = np.std(img_array, axis=(0, 1))

            # Perform normalization for saving if output_path is provided
            if output_path:
                epsilon = 1e-6
                normalized_array = (img_array - mean) / (std + epsilon)
                min_val, max_val = np.min(normalized_array), np.max(normalized_array)
                scaled_array = 255 * (normalized_array - min_val) / (max_val - min_val)
                scaled_array = scaled_array.astype(np.uint8)
                normalized_img = Image.fromarray(scaled_array)
                normalized_img.save(output_path, format=image_format)

            return {
                "mean": np.round(mean, 2).tolist(),
                "std": np.round(std, 2).tolist(),
            }

    def normalize_image_from_bytes(
        self, image_bytes: bytes, image_format: str
    ) -> bytes:
        """
        Normalize an image from bytes.
        The normalization is a contrast stretch for visualization.

        Args:
            image_bytes: Image data as bytes
            image_format: The format of the image (e.g., 'jpeg', 'png')

        Returns:
            Normalized image as bytes
        """
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img).astype(np.float32)

        # Use a small epsilon to avoid division by zero
        epsilon = 1e-6
        mean = np.mean(img_array, axis=(0, 1))
        std = np.std(img_array, axis=(0, 1))
        normalized_array = (img_array - mean) / (std + epsilon)

        # Scale to 0-255 for visualization as a standard image
        min_val, max_val = np.min(normalized_array), np.max(normalized_array)
        scaled_array = 255 * (normalized_array - min_val) / (max_val - min_val)
        scaled_array = scaled_array.astype(np.uint8)

        normalized_img = Image.fromarray(scaled_array)
        output_bytes = io.BytesIO()
        normalized_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()

    def crop_image(
        self,
        image_path: str,
        box: Tuple[int, int, int, int],
        output_path: str = None,
    ) -> Tuple[int, int]:
        """
        Crop an image to specified coordinates.

        Args:
            image_path: Path to input image
            box: A tuple of (left, top, right, bottom)
            output_path: Path to save cropped image (optional)

        Returns:
            Tuple of (width, height) of cropped image
        """
        with Image.open(image_path) as img:
            cropped_img = img.crop(box)

            if output_path:
                cropped_img.save(output_path)

            return cropped_img.size

    def crop_image_from_bytes(
        self, image_bytes: bytes, box: Tuple[int, int, int, int], image_format: str
    ) -> bytes:
        """
        Crop an image from bytes.

        Args:
            image_bytes: Image data as bytes
            box: A tuple of (left, top, right, bottom)
            image_format: The format of the image (e.g., 'jpeg', 'png')

        Returns:
            Cropped image as bytes
        """
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop(box)
        output_bytes = io.BytesIO()
        cropped_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()
