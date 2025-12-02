# MLOps Lab 3: Experiment Tracking and Model Versioning

This project is the final stage of the MLOps Laboratory series, building upon the API/CLI foundation of **Lab 1** and the Docker/CD pipeline of **Lab 2**.

In this lab, we replace the random prediction logic with a real Deep Learning model (**MobileNet_v2**) trained on the Oxford-IIIT Pet dataset. We demonstrate how to track experiments and version models using **MLFlow**, and how to deploy a serialized **ONNX** model using **FastAPI** and **Docker**.

## Project Structure

- `src/train.py`: Script to train MobileNet_v2 on the Oxford-IIIT Pet dataset. Logs parameters, metrics, and artifacts to MLFlow.
- `src/select_model.py`: Script to query MLFlow for the best performing model, export it to ONNX format, and download class labels.
- `src/inference.py`: Standalone script to demonstrate inference using the exported ONNX model.
- `logic/predictor.py`: Logic class used by the API to load the ONNX model and perform predictions.
- `api/api.py`: FastAPI application serving the model.
- `Dockerfile`: Configuration to containerize the application.

## Setup

1. **Install dependencies**:
   This project uses `uv` for dependency management.
   ```bash
   uv sync
   ```

2. **Activate environment**:
   ```bash
   source .venv/bin/activate
   ```

## Workflow

### 1. Train the Model
Train the model using transfer learning (MobileNet_v2). You can run multiple experiments with different configurations.
```bash
uv run python src/train.py --epochs 5 --batch_size 32 --learning_rate 0.001
```

### 2. Track Experiments
Visualize your experiments using the MLFlow UI.
```bash
uv run mlflow ui
```
Open your browser at `http://127.0.0.1:5000`.

### 3. Select and Export Best Model
Select the best model based on validation accuracy and export it to ONNX.
```bash
uv run python src/select_model.py
```
This will create `results/model.onnx` and `results/classes.json`.

### 4. Run the API Locally
Start the FastAPI server to serve predictions.
```bash
uv run uvicorn api.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` to test the `/predict` endpoint.

### 5. Docker
Build and run the container.
```bash
docker build -t mlops-lab3 .
docker run -p 8000:8000 mlops-lab3
```

## Testing
Run tests to verify artifacts existence.
```bash
uv run pytest tests/
```
