"""
Gradio application for the Image Classification API.

This application provides a user-friendly interface to interact with the FastAPI backend.
It is hosted on Hugging Face Spaces and communicates with the API deployed on Render.
"""

import gradio as gr
import requests
import os
from PIL import Image
import io

# URL of the API hosted in Render
# We strip trailing slashes to ensure correct URL construction
API_URL = os.getenv("API_URL", "https://your-render-service.onrender.com").rstrip("/")
print(f"🚀 Using API URL: {API_URL}")

def predict(image):
    """
    Send an image to the API for classification.
    """
    if image is None:
        return "No image provided"
    try:
        with open(image, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_URL}/predict", files=files)
        if response.status_code == 200:
            data = response.json()
            return f"Class: {data['predicted_class']} (Confidence: {data['confidence']})"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def resize(image, width, height):
    """
    Send an image to the API for resizing.
    """
    if image is None:
        return None
    try:
        with open(image, "rb") as f:
            files = {"file": f}
            params = {"width": int(width), "height": int(height)}
            response = requests.post(f"{API_URL}/resize", files=files, params=params)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception:
        return None

def grayscale(image):
    """
    Send an image to the API for grayscale conversion.
    """
    if image is None:
        return None
    try:
        with open(image, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_URL}/grayscale", files=files)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception:
        return None

def crop(image, left, top, right, bottom):
    """
    Send an image to the API for cropping.
    """
    if image is None:
        return None
    try:
        with open(image, "rb") as f:
            files = {"file": f}
            params = {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)}
            response = requests.post(f"{API_URL}/crop", files=files, params=params)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception:
        return None

def normalize(image):
    """
    Send an image to the API for normalization.
    """
    if image is None:
        return None
    try:
        with open(image, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_URL}/normalize", files=files)
        if response.status_code == 200:
            # Normalize returns an image in this API implementation
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception:
        return None

# Define the Gradio Interface with Tabs
with gr.Blocks(title="MLOps Lab 2 - Image Tools") as app:
    gr.Markdown("# 🖼️ Image Processing & Classification API")
    gr.Markdown("Upload an image and choose a tool below.")

    # Tab 1: Prediction
    with gr.Tab("🔮 Predict"):
        with gr.Row():
            pred_input = gr.Image(type="filepath", label="Upload Image")
            pred_output = gr.Textbox(label="Prediction Result")
        pred_button = gr.Button("Predict Class")
        pred_button.click(predict, inputs=pred_input, outputs=pred_output)

    # Tab 2: Resize
    with gr.Tab("📏 Resize"):
        with gr.Row():
            resize_input = gr.Image(type="filepath", label="Upload Image")
            with gr.Column():
                width_input = gr.Number(value=224, label="Width")
                height_input = gr.Number(value=224, label="Height")
            resize_output = gr.Image(label="Resized Image")
        resize_button = gr.Button("Resize Image")
        resize_button.click(resize, inputs=[resize_input, width_input, height_input], outputs=resize_output)

    # Tab 3: Grayscale
    with gr.Tab("⚫ Grayscale"):
        with gr.Row():
            gray_input = gr.Image(type="filepath", label="Upload Image")
            gray_output = gr.Image(label="Grayscale Image")
        gray_button = gr.Button("Convert to Grayscale")
        gray_button.click(grayscale, inputs=gray_input, outputs=gray_output)

    # Tab 4: Crop
    with gr.Tab("✂️ Crop"):
        with gr.Row():
            crop_input = gr.Image(type="filepath", label="Upload Image")
            with gr.Column():
                left_input = gr.Number(value=0, label="Left")
                top_input = gr.Number(value=0, label="Top")
                right_input = gr.Number(value=200, label="Right")
                bottom_input = gr.Number(value=200, label="Bottom")
            crop_output = gr.Image(label="Cropped Image")
        crop_button = gr.Button("Crop Image")
        crop_button.click(crop, inputs=[crop_input, left_input, top_input, right_input, bottom_input], outputs=crop_output)

    # Tab 5: Normalize
    with gr.Tab("📊 Normalize"):
        with gr.Row():
            norm_input = gr.Image(type="filepath", label="Upload Image")
            norm_output = gr.Image(label="Normalized Image")
        norm_button = gr.Button("Normalize Image")
        norm_button.click(normalize, inputs=norm_input, outputs=norm_output)

if __name__ == "__main__":
    app.launch()
