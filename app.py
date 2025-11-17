import gradio as gr
import requests

# URL of the API created with FastAPI
# API_URL = "https://my-fastapi-main-latest.onrender.com"
API_URL = "https://calculator-latest.onrender.com"


# Function to execute when clicking the "Compute button"
def calcular(a, b, operacion):
    try:
        payload = {"a": float(a), "b": float(b), "op": operacion}
        response = requests.post(f"{API_URL}/calculate", json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("result")
    except requests.exceptions.HTTPError as e:
        return f"Error: {response.json().get('detail', str(e))}"


# GUI creted using Gradio
iface = gr.Interface(
    fn=calcular,
    inputs=[
        gr.Number(label="Number X"),
        gr.Number(label="Number Y"),
        gr.Dropdown(
            choices=["add", "subtract", "multiply", "divide", "power"],
            value="add",
            label="Arithmetical operation",
        ),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Calculator created with FastAPI and Gradio",
    description="Interactive calculator using the endpoint /calculate",
)

# Launch the GUI
if __name__ == "__main__":
    iface.launch()
