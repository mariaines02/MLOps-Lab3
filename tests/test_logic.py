"""
Unit tests for the image predictor logic.
"""

import pytest
from PIL import Image
import io
import os
from logic.predictor import ImagePredictor


@pytest.fixture
def predictor():
    """Fixture providing an ImagePredictor instance."""
    return ImagePredictor()


@pytest.fixture
def custom_predictor():
    """Fixture providing a predictor with custom classes."""
    return ImagePredictor(class_names=["apple", "banana", "orange"])


@pytest.fixture
def sample_image(tmp_path):
    """Fixture creating a sample image for testing."""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_image_bytes():
    """Fixture providing sample image as bytes."""
    img = Image.new("RGB", (100, 100), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def test_predictor_initialization(predictor):
    """Test predictor initializes with default classes."""
    assert len(predictor.class_names) == 10
    assert "cat" in predictor.class_names
    assert "dog" in predictor.class_names


def test_custom_predictor_initialization(custom_predictor):
    """Test predictor initializes with custom classes."""
    assert len(custom_predictor.class_names) == 3
    assert "apple" in custom_predictor.class_names


def test_predict_returns_dict(predictor):
    """Test predict returns a dictionary with expected keys."""
    result = predictor.predict()
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "all_classes" in result


def test_predict_with_seed_reproducibility(predictor):
    """Test predict with seed produces reproducible results."""
    result1 = predictor.predict(seed=42)
    result2 = predictor.predict(seed=42)
    assert result1["predicted_class"] == result2["predicted_class"]
    assert result1["confidence"] == result2["confidence"]


def test_predict_confidence_range(predictor):
    """Test predict confidence is within expected range."""
    result = predictor.predict(seed=123)
    assert 0.7 <= result["confidence"] <= 0.99


def test_predict_class_in_list(predictor):
    """Test predicted class is from the class list."""
    result = predictor.predict(seed=456)
    assert result["predicted_class"] in predictor.class_names


def test_resize_image(predictor, sample_image):
    """Test image resizing functionality."""
    new_size = predictor.resize_image(sample_image, 50, 50)
    assert new_size == (50, 50)


def test_resize_image_with_output(predictor, sample_image, tmp_path):
    """Test image resizing with output file."""
    output_path = tmp_path / "resized.jpg"
    new_size = predictor.resize_image(sample_image, 200, 200, str(output_path))
    assert new_size == (200, 200)
    assert output_path.exists()

    # Verify the saved image
    with Image.open(output_path) as img:
        assert img.size == (200, 200)


def test_resize_image_from_bytes(predictor, sample_image_bytes):
    """Test resizing image from bytes."""
    resized_bytes = predictor.resize_image_from_bytes(
        sample_image_bytes, 50, 50, image_format="PNG"
    )
    assert isinstance(resized_bytes, bytes)

    with Image.open(io.BytesIO(resized_bytes)) as img:
        assert img.size == (50, 50)


def test_convert_to_grayscale(predictor, sample_image):
    """Test grayscale conversion."""
    mode = predictor.convert_to_grayscale(sample_image)
    assert mode == "L"


def test_convert_to_grayscale_with_output(predictor, sample_image, tmp_path):
    """Test grayscale conversion with output file."""
    output_path = tmp_path / "grayscale.jpg"
    mode = predictor.convert_to_grayscale(sample_image, str(output_path))
    assert mode == "L"
    assert output_path.exists()

    # Verify the saved image
    with Image.open(output_path) as img:
        assert img.mode == "L"


def test_normalize_image(predictor, sample_image):
    """Test image normalization statistics."""
    stats = predictor.normalize_image(sample_image)
    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "std" in stats


def test_normalize_image_values(predictor, sample_image):
    """Test normalization values are within expected ranges."""
    stats = predictor.normalize_image(sample_image)
    # Check that mean for each channel is within the valid range
    for val in stats["mean"]:
        assert 0 <= val <= 255


def test_normalize_single_color_image(predictor):
    """Test normalization of a single-color image to avoid division by zero."""
    img = Image.new("RGB", (10, 10), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    normalized_bytes = predictor.normalize_image_from_bytes(
        img_bytes.getvalue(), "jpeg"
    )
    assert isinstance(normalized_bytes, bytes)
    # The output should be a valid image (e.g., all black if std is zero)
    assert len(normalized_bytes) > 0


def test_crop_image(predictor, sample_image):
    """Test image cropping."""
    cropped_size = predictor.crop_image(sample_image, (10, 10, 60, 60))
    assert cropped_size == (50, 50)


def test_crop_image_with_output(predictor, sample_image, tmp_path):
    """Test image cropping with output file."""
    output_path = tmp_path / "cropped.jpg"
    cropped_size = predictor.crop_image(
        sample_image, (20, 20, 80, 80), str(output_path)
    )
    assert cropped_size == (60, 60)
    assert output_path.exists()

    # Verify the saved image
    with Image.open(output_path) as img:
        assert img.size == (60, 60)


def test_resize_invalid_image_path(predictor):
    """Test resize with invalid image path."""
    with pytest.raises(FileNotFoundError):
        predictor.resize_image("nonexistent.jpg", 100, 100)


def test_grayscale_invalid_image_path(predictor):
    """Test grayscale with invalid image path."""
    with pytest.raises(FileNotFoundError):
        predictor.convert_to_grayscale("nonexistent.jpg")


def test_normalize_invalid_image_path(predictor):
    """Test normalize with invalid image path."""
    with pytest.raises(FileNotFoundError):
        predictor.normalize_image("nonexistent.jpg")
