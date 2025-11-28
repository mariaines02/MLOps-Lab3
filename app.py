import gradio as gr
import requests
import os

# URL of the API hosted in Render
# The user should replace this with their actual Render URL
# Example: https://mlops-lab2-demo.onrender.com
API_URL = os.getenv("API_URL", "https://your-render-service.onrender.com")

def predict(image):
    if image is None:
        return "No image provided"
    
    try:
        # Gradio passes the image path when type="filepath"
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

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Image Classifier",
    description="Upload an image to get a random class prediction from the API."
)

if __name__ == "__main__":
    iface.launch()
