"""
FastAPI application for image prediction and preprocessing.
"""

import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from logic.predictor import ImagePredictor


app = FastAPI(
    title="Image Classification API",
    description="API for image classification and preprocessing",
    version="1.0.0",
)

predictor = ImagePredictor()


class PredictionRequest(BaseModel):
    """Request model for prediction."""

    seed: int | None = None


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    predicted_class: str
    confidence: float
    all_classes: list[str]


class ResizeRequest(BaseModel):
    """Request model for image resizing."""

    width: int
    height: int


class ResizeResponse(BaseModel):
    """Response model for image resizing."""

    original_size: tuple[int, int]
    new_size: tuple[int, int]
    message: str


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Home endpoint serving the main page.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classification API</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.95);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                color: #333;
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 40px;
                font-size: 1.2em;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px;
                border-radius: 15px;
                color: white;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
            }
            .feature-card h3 {
                margin-top: 0;
                font-size: 1.3em;
            }
            .cta-button {
                display: block;
                background: #667eea;
                color: white;
                padding: 15px 30px;
                text-align: center;
                text-decoration: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 1.1em;
                margin: 30px auto;
                max-width: 300px;
                transition: background 0.3s ease;
            }
            .cta-button:hover {
                background: #764ba2;
            }
            .info-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #667eea;
            }
            .endpoint-list {
                list-style: none;
                padding: 0;
            }
            .endpoint-list li {
                padding: 10px;
                margin: 5px 0;
                background: white;
                border-radius: 5px;
                border-left: 3px solid #667eea;
            }
            code {
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🖼️ Image Classification API</h1>
            <p class="subtitle">Machine Learning powered image processing and classification</p>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>🎯 Prediction</h3>
                    <p>Classify images into predefined categories with confidence scores.</p>
                </div>
                <div class="feature-card">
                    <h3>📏 Resize</h3>
                    <p>Resize images to any dimensions while maintaining quality.</p>
                </div>
                <div class="feature-card">
                    <h3>⚫ Grayscale</h3>
                    <p>Convert colored images to grayscale for processing.</p>
                </div>
                <div class="feature-card">
                    <h3>✂️ Crop</h3>
                    <p>Crop images to specific regions of interest.</p>
                </div>
            </div>

            <a href="/docs" class="cta-button">📚 Explore API Documentation</a>

            <div class="info-section">
                <h2>Available Endpoints</h2>
                <ul class="endpoint-list">
                    <li><strong>POST /predict</strong> - Predict image class</li>
                    <li><strong>POST /resize</strong> - Resize an image</li>
                    <li><strong>POST /grayscale</strong> - Convert to grayscale</li>
                    <li><strong>POST /normalize</strong> - Get image statistics</li>
                    <li><strong>POST /crop</strong> - Crop an image</li>
                    <li><strong>GET /health</strong> - Check API health</li>
                </ul>
            </div>

            <div class="info-section">
                <h2>Quick Start</h2>
                <p>1. Visit <code>/docs</code> for interactive API documentation</p>
                <p>2. Use the "Try it out" button on any endpoint</p>
                <p>3. Upload an image and see the results instantly</p>
            </div>

            <p style="text-align: center; margin-top: 40px; color: #666;">
                <strong>MLOps Lab1</strong> - Continuous Integration with GitHub Actions<br>
                Version 1.0.0 | Built with FastAPI & PIL
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "Image Classification API",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...), seed: int | None = None
):  # pylint: disable=unused-argument
    """
    Predict the class of an uploaded image.

    Args:
        file: Image file to classify
        seed: Optional random seed for reproducibility

    Returns:
        Prediction results with class and confidence
    """
    try:
        # For now, prediction is random (will be replaced with real model in Lab3)
        result = predictor.predict(image_path=file.filename, seed=seed)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...), width: int = 224, height: int = 224
):
    """
    Resize an uploaded image.

    Args:
        file: Image file to resize
        width: Target width (default: 224)
        height: Target height (default: 224)

    Returns:
        Resized image
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        resized_bytes = predictor.resize_image_from_bytes(
            contents, width, height, image_format
        )

        return StreamingResponse(
            io.BytesIO(resized_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=resized_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/grayscale")
async def convert_grayscale(file: UploadFile = File(...)):
    """
    Convert an uploaded image to grayscale.

    Args:
        file: Image file to convert

    Returns:
        Grayscale image
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        output_bytes = predictor.convert_to_grayscale_from_bytes(contents, image_format)

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=gray_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    left: int = 0,
    top: int = 0,
    right: int = 224,
    bottom: int = 224,
):
    """
    Crop an uploaded image.

    Args:
        file: Image file to crop
        left: Left coordinate
        top: Top coordinate
        right: Right coordinate
        bottom: Bottom coordinate

    Returns:
        Cropped image
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        box = (left, top, right, bottom)
        output_bytes = predictor.crop_image_from_bytes(contents, box, image_format)

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=cropped_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/normalize")
async def get_image_stats(file: UploadFile = File(...)):
    """
    Normalize an uploaded image.

    Args:
        file: Image file to analyze

    Returns:
        Normalized image
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        output_bytes = predictor.normalize_image_from_bytes(contents, image_format)

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=normalized_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
