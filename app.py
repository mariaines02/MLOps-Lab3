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
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

css = """
.container { max-width: 1200px; margin: auto; }
.header { text-align: center; margin-bottom: 2rem; }
"""

with gr.Blocks(title="MLOps Lab 3 - Pet Classifier") as app:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            # 🐾 Pet Classifier & Image Tools
            ### MLOps Lab 3 Demo
            Upload an image to classify the pet breed or perform image processing operations.
            """,
            elem_classes="header",
        )

        with gr.Tabs():
            # Tab 1: Prediction
            with gr.TabItem("🔮 Predict"):
                with gr.Row():
                    with gr.Column():
                        pred_input = gr.Image(type="filepath", label="Upload Image", height=400)
                        pred_button = gr.Button("Predict Class", variant="primary", size="lg")
                        
                        gr.Examples(
                            examples=["test_image.jpg"],
                            inputs=pred_input,
                            label="Try an example"
                        )

                    with gr.Column():
                        pred_output = gr.Textbox(
                            label="Prediction Result",
                            lines=4,                    
                        )
                
                pred_button.click(predict, inputs=pred_input, outputs=pred_output)

            # Tab 2: Resize
            with gr.TabItem("📏 Resize"):
                with gr.Row():
                    with gr.Column():
                        resize_input = gr.Image(type="filepath", label="Upload Image", height=400)
                        with gr.Row():
                            width_input = gr.Number(value=224, label="Width", precision=0)
                            height_input = gr.Number(value=224, label="Height", precision=0)
                        resize_button = gr.Button("Resize Image", variant="primary")
                    
                    with gr.Column():
                        resize_output = gr.Image(label="Resized Image")
                
                resize_button.click(resize, inputs=[resize_input, width_input, height_input], outputs=resize_output)

            # Tab 3: Grayscale
            with gr.TabItem("⚫ Grayscale"):
                with gr.Row():
                    with gr.Column():
                        gray_input = gr.Image(type="filepath", label="Upload Image", height=400)
                        gray_button = gr.Button("Convert to Grayscale", variant="primary")
                    
                    with gr.Column():
                        gray_output = gr.Image(label="Grayscale Image")
                
                gray_button.click(grayscale, inputs=gray_input, outputs=gray_output)

            # Tab 4: Crop
            with gr.TabItem("✂️ Crop"):
                with gr.Row():
                    with gr.Column():
                        crop_input = gr.Image(type="filepath", label="Upload Image", height=400)
                        with gr.Row():
                            left_input = gr.Number(value=0, label="Left", precision=0)
                            top_input = gr.Number(value=0, label="Top", precision=0)
                            right_input = gr.Number(value=200, label="Right", precision=0)
                            bottom_input = gr.Number(value=200, label="Bottom", precision=0)
                        crop_button = gr.Button("Crop Image", variant="primary")
                    
                    with gr.Column():
                        crop_output = gr.Image(label="Cropped Image")
                
                crop_button.click(crop, inputs=[crop_input, left_input, top_input, right_input, bottom_input], outputs=crop_output)

            # Tab 5: Normalize
            with gr.TabItem("📊 Normalize"):
                with gr.Row():
                    with gr.Column():
                        norm_input = gr.Image(type="filepath", label="Upload Image", height=400)
                        norm_button = gr.Button("Normalize Image", variant="primary")
                    
                    with gr.Column():
                        norm_output = gr.Image(label="Normalized Image")
                
                norm_button.click(normalize, inputs=norm_input, outputs=norm_output)

if __name__ == "__main__":
    app.launch()
