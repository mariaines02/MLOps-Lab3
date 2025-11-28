"""
Machine Learning logic for image classification and preprocessing.
"""

import io
import random
from typing import Tuple, List
import numpy as np
from PIL import Image


class ImagePredictor:
    """Class for image prediction and preprocessing."""

    def __init__(self, class_names: List[str] = None):
        """
        Initialize the predictor with class names.

        Args:
            class_names: List of possible class names for prediction
        """
        if class_names is None:
            self.class_names = [
                "cat",
                "dog",
                "bird",
                "fish",
                "horse",
                "car",
                "bicycle",
                "airplane",
                "boat",
                "train",
            ]
        else:
            self.class_names = class_names

    def predict(  # pylint: disable=unused-argument
        self, image_path: str = None, seed: int = None
    ) -> dict:
        """
        Predict the class of an image (randomly for now).

        Args:
            image_path: Path to the image file
            seed: Random seed for reproducibility

        Returns:
            Dictionary with prediction results
        """
        if seed is not None:
            random.seed(seed)

        predicted_class = random.choice(self.class_names)
        confidence = round(random.uniform(0.7, 0.99), 2)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_classes": self.class_names,
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
