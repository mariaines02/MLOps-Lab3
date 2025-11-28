"""
Command Line Interface for image prediction and preprocessing.
"""

import click
from logic.predictor import ImagePredictor


predictor = ImagePredictor()


@click.group()
def cli():
    """Image classification and preprocessing CLI."""


@cli.command()
@click.option("--image", "-i", default=None, help="Path to image file")
@click.option(
    "--seed", "-s", default=None, type=int, help="Random seed for reproducibility"
)
def predict(image, seed):
    """
    Predict the class of an image.

    Example: python -m cli.cli predict --image photo.jpg --seed 42
    """
    result = predictor.predict(image_path=image, seed=seed)
    click.echo(f"Predicted class: {result['predicted_class']}")
    click.echo(f"Confidence: {result['confidence']}")


@cli.command()
@click.argument("image_path")
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option("--output", "-o", default=None, help="Output path for resized image")
def resize(image_path, width, height, output):
    """
    Resize an image to specified dimensions.

    Example: python -m cli.cli resize input.jpg 224 224 --output resized.jpg
    """
    try:
        new_size = predictor.resize_image(image_path, width, height, output)
        click.echo(f"Image resized to: {new_size[0]}x{new_size[1]}")
        if output:
            click.echo(f"Saved to: {output}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except (IOError, ValueError) as e:
        click.echo(f"Error processing image: {e}", err=True)


@cli.command()
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path for grayscale image")
def grayscale(image_path, output):
    """
    Convert an image to grayscale.

    Example: python -m cli.cli grayscale input.jpg --output gray.jpg
    """
    try:
        mode = predictor.convert_to_grayscale(image_path, output)
        click.echo(f"Image converted to grayscale (mode: {mode})")
        if output:
            click.echo(f"Saved to: {output}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except IOError as e:
        click.echo(f"Error processing image: {e}", err=True)


@cli.command()
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path for normalized image")
def normalize(image_path, output):
    """
    Normalize an image and get its statistics.

    Example: python -m cli.cli normalize input.jpg --output normalized.jpg
    """
    try:
        stats = predictor.normalize_image(image_path, output)
        click.echo("Image normalized.")
        click.echo(f"  Mean: {stats['mean']}")
        click.echo(f"  Std Dev: {stats['std']}")
        if output:
            click.echo(f"Saved to: {output}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except IOError as e:
        click.echo(f"Error reading image: {e}", err=True)


@cli.command()
@click.argument("image_path")
@click.option(
    "--box",
    nargs=4,
    type=int,
    required=True,
    help="A tuple of (left, top, right, bottom)",
)
@click.option("--output", "-o", default=None, help="Output path for cropped image")
def crop(image_path, box, output):
    """
    Crop an image to specified coordinates.

    Example: python -m cli.cli crop input.jpg --box 10 10 200 200 --output cropped.jpg
    """
    try:
        size = predictor.crop_image(image_path, box, output)
        click.echo(f"Image cropped to: {size[0]}x{size[1]}")
        if output:
            click.echo(f"Saved to: {output}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except (IOError, ValueError) as e:
        click.echo(f"Error cropping image: {e}", err=True)


if __name__ == "__main__":
    cli()
