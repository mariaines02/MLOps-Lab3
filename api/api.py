"""
FastAPI application for image prediction and preprocessing.
"""

import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from logic.predictor import ImagePredictor


app = FastAPI(
    title="MLOps Lab API",
    description="API for image classification (Lab 2)",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")

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
async def home(request: Request):
    """
    Home endpoint serving the main page.
    """
    return templates.TemplateResponse(request, "home.html")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "MLOps Lab API",
        "version": "1.0.0",
    }


#  Image Classification Endpoints ---


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
