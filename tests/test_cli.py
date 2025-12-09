"""
Integration tests for the CLI module.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner
from PIL import Image
from cli.cli import cli


@pytest.fixture
def runner():
    """Fixture to create a CliRunner instance."""
    return CliRunner()


@pytest.fixture
def sample_image(tmp_path):
    """Fixture creating a sample image for CLI testing."""
    img_path = tmp_path / "test_cli_image.jpg"
    img = Image.new("RGB", (100, 100), color="green")
    img.save(img_path)
    return str(img_path)


def test_cli_predict_command(runner):
    """Test CLI predict command."""
    result = runner.invoke(cli, ["predict", "--seed", "42"])
    assert result.exit_code == 0
    assert "Predicted class:" in result.output
    assert "Confidence:" in result.output


def test_cli_predict_with_seed_reproducibility(runner):
    """Test CLI predict command reproducibility with seed."""
    result1 = runner.invoke(cli, ["predict", "--seed", "100"])
    result2 = runner.invoke(cli, ["predict", "--seed", "100"])
    assert result1.output == result2.output


def test_cli_predict_with_image_path(runner, sample_image):
    """Test CLI predict command with image path."""
    result = runner.invoke(cli, ["predict", "--image", sample_image, "--seed", "42"])
    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_cli_resize_command(runner, sample_image, tmp_path):
    """Test CLI resize command."""
    output_path = tmp_path / "resized_cli.jpg"
    result = runner.invoke(
        cli, ["resize", sample_image, "50", "50", "--output", str(output_path)]
    )
    assert result.exit_code == 0
    assert "Image resized to: 50x50" in result.output
    assert output_path.exists()


def test_cli_resize_without_output(runner, sample_image):
    """Test CLI resize command without output path."""
    result = runner.invoke(cli, ["resize", sample_image, "75", "75"])
    assert result.exit_code == 0
    assert "Image resized to: 75x75" in result.output


def test_cli_resize_invalid_image(runner):
    """Test CLI resize with invalid image path."""
    result = runner.invoke(cli, ["resize", "nonexistent.jpg", "100", "100"])
    assert result.exit_code == 0  # CLI handles error gracefully
    assert "Error" in result.output


@patch("logic.predictor.ImagePredictor.resize_image")
def test_cli_resize_handles_io_error(mock_resize, runner, sample_image):
    """Test CLI resize handles IO errors."""
    mock_resize.side_effect = IOError("Disk full")
    result = runner.invoke(cli, ["resize", sample_image, "100", "100"])
    assert "Error processing image: Disk full" in result.output


def test_cli_grayscale_command(runner, sample_image, tmp_path):
    """Test CLI grayscale command."""
    output_path = tmp_path / "gray_cli.jpg"
    result = runner.invoke(
        cli, ["grayscale", sample_image, "--output", str(output_path)]
    )
    assert result.exit_code == 0
    assert "Image converted to grayscale" in result.output
    assert output_path.exists()


def test_cli_grayscale_without_output(runner, sample_image):
    """Test CLI grayscale command without output path."""
    result = runner.invoke(cli, ["grayscale", sample_image])
    assert result.exit_code == 0
    assert "Image converted to grayscale" in result.output


@patch("logic.predictor.ImagePredictor.convert_to_grayscale")
def test_cli_grayscale_handles_io_error(mock_grayscale, runner, sample_image):
    """Test CLI grayscale handles IO errors."""
    mock_grayscale.side_effect = IOError("Permission denied")
    result = runner.invoke(cli, ["grayscale", sample_image])
    assert "Error processing image: Permission denied" in result.output


def test_cli_normalize_command(runner, sample_image):
    """Test CLI normalize command."""
    result = runner.invoke(cli, ["normalize", sample_image])
    assert result.exit_code == 0
    assert "Image normalized." in result.output
    assert "Mean:" in result.output
    assert "Std Dev:" in result.output


def test_cli_normalize_with_output(runner, sample_image, tmp_path):
    """Test CLI normalize command with output."""
    output_path = tmp_path / "normalized_cli.jpg"
    result = runner.invoke(
        cli, ["normalize", sample_image, "--output", str(output_path)]
    )
    assert result.exit_code == 0
    assert "Image normalized." in result.output
    assert output_path.exists()


def test_cli_normalize_invalid_image(runner):
    """Test CLI normalize with invalid image."""
    result = runner.invoke(cli, ["normalize", "nonexistent.jpg"])
    assert result.exit_code == 0
    assert "Error" in result.output


@patch("logic.predictor.ImagePredictor.normalize_image")
def test_cli_normalize_handles_io_error(mock_normalize, runner, sample_image):
    """Test CLI normalize handles IO errors."""
    mock_normalize.side_effect = IOError("Cannot read")
    result = runner.invoke(cli, ["normalize", sample_image])
    assert "Error reading image: Cannot read" in result.output


def test_cli_crop_command(runner, sample_image, tmp_path):
    """Test CLI crop command."""
    output_path = tmp_path / "cropped_cli.jpg"
    result = runner.invoke(
        cli,
        [
            "crop",
            sample_image,
            "--box",
            "10",
            "10",
            "60",
            "60",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert "Image cropped to:" in result.output
    assert output_path.exists()


def test_cli_crop_without_output(runner, sample_image):
    """Test CLI crop command without output path."""
    result = runner.invoke(cli, ["crop", sample_image, "--box", "20", "20", "80", "80"])
    assert result.exit_code == 0
    assert "Image cropped to:" in result.output


def test_cli_crop_invalid_image(runner):
    """Test CLI crop with invalid image path."""
    result = runner.invoke(
        cli, ["crop", "nonexistent.jpg", "--box", "0", "0", "1", "1"]
    )
    assert result.exit_code == 0
    assert "Error" in result.output


@patch("logic.predictor.ImagePredictor.crop_image")
def test_cli_crop_handles_value_error(mock_crop, runner, sample_image):
    """Test CLI crop handles Value errors."""
    mock_crop.side_effect = ValueError("Bad coordinates")
    result = runner.invoke(cli, ["crop", sample_image, "--box", "0", "0", "1", "1"])
    assert "Error cropping image: Bad coordinates" in result.output


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Image classification and preprocessing CLI" in result.output


def test_cli_predict_help(runner):
    """Test CLI predict help."""
    result = runner.invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0
    assert "Predict the class of an image" in result.output


def test_cli_resize_help(runner):
    """Test CLI resize help."""
    result = runner.invoke(cli, ["resize", "--help"])
    assert result.exit_code == 0
    assert "Resize an image" in result.output
