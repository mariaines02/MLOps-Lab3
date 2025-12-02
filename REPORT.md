# MLOps Lab 3 Report

## Repositories & Spaces
- **GitHub Repository (Lab 1)**: [Link to Lab 1 Repo]
- **GitHub Repository (Lab 2)**: [Link to Lab 2 Repo]
- **GitHub Repository (Lab 3)**: [Link to Lab 3 Repo]
- **HuggingFace Space (Lab 2)**: [Link to Lab 2 Space]
- **HuggingFace Space (Lab 3)**: [Link to Lab 3 Space]

## Project Evolution
This project represents the culmination of a three-part MLOps laboratory series, building directly upon the foundations laid in Lab 1 and Lab 2.

### Lab 1: Foundation (CI & Logic)
-   **Goal**: Establish the core application structure and Continuous Integration.
-   **Contribution**: Created the `ImagePredictor` class (initially with random predictions), the FastAPI backend (`api/`), and the CLI (`cli/`). Implemented image preprocessing (resize, crop, grayscale) and set up a GitHub Actions CI pipeline for linting and testing.

### Lab 2: Deployment (CD & Containerization)
-   **Goal**: Automate deployment and containerize the application.
-   **Contribution**: Dockerized the application (`Dockerfile`). Implemented Continuous Delivery to deploy the API to Render and the UI (Gradio) to Hugging Face Spaces. Added `app.py` for the frontend interface.

### Lab 3: Intelligence (MLOps & Model Management)
-   **Goal**: Replace the random predictor with a real, trained machine learning model.
-   **Contribution**:
    -   Integrated **MLFlow** for experiment tracking (`src/train.py`).
    -   Implemented model versioning and selection (`src/select_model.py`).
    -   Replaced the random logic in `ImagePredictor` with an **ONNX** runtime inference engine using a MobileNet_v2 model trained on the Oxford-IIIT Pet dataset.
    -   Ensured the entire pipeline (Training -> Selection -> Inference) is reproducible and automated.

## Testing Logic
The project employs a comprehensive testing strategy covering three main components:
1.  **Logic Tests (`tests/test_logic.py`)**:
    -   Verifies the `ImagePredictor` class.
    -   Tests initialization (loading class labels).
    -   Tests prediction logic (using the ONNX model).
    -   Tests preprocessing functions (resize, grayscale, normalize, crop).
    -   Ensures reproducibility by checking that the same seed yields the same result.
2.  **API Tests (`tests/test_api.py`)**:
    -   Integration tests for the FastAPI application.
    -   Verifies all endpoints (`/predict`, `/resize`, etc.) return correct status codes and data structures.
    -   Checks error handling for invalid inputs.
3.  **Artifact Tests (`tests/test_artifacts.py`)**:
    -   Ensures that the necessary artifacts (`results/model.onnx` and `results/classes.json`) are present before containerization.
    -   This is critical for the Docker build to succeed and the application to function.

## Experiments Conducted
We used **MLFlow** to track experiments for training a **MobileNet_v2** model on the **Oxford-IIIT Pet** dataset.

### Experimental Setup
-   **Model**: MobileNet_v2 (pretrained on ImageNet).
-   **Transfer Learning**: Frozen feature extractor, modified classifier head (Linear layer with 37 outputs).
-   **Optimizer**: Adam.
-   **Loss Function**: CrossEntropyLoss.

### Logged Artifacts
For each run, we logged:
1.  **Parameters**: `batch_size`, `learning_rate`, `epochs`, `seed`, `optimizer`, `loss_function`.
2.  **Metrics**: `train_loss`, `train_acc`, `val_loss`, `val_acc` (logged per epoch).
3.  **Artifacts**:
    -   `classes.json`: The list of class names to ensure correct mapping during inference.
    -   `training_curves.png`: A plot of loss and accuracy curves to visually assess training progress and overfitting.
4.  **Model**: The trained PyTorch model, registered as `oxford_pet_classifier`.

## Results Analysis
Using the MLFlow UI, we analyzed the runs to select the best model.

-   **Selection Criteria**: We selected the model with the highest **Validation Accuracy (`val_acc`)**.
-   **Best Run**:
    -   **Run ID**: [Insert Run ID from select_model.py output]
    -   **Validation Accuracy**: [Insert Best Val Acc]
    -   **Configuration**: Batch Size = [Value], Learning Rate = [Value].

The selected model was then loaded, converted to **ONNX** format (opset 18) for optimized inference, and saved to `results/model.onnx`. This serialized model is used by the API for production inference.

## Conclusion
This project successfully demonstrates a complete MLOps lifecycle, from experimentation to deployment. By integrating MLFlow, we achieved full traceability of our experiments, allowing for data-driven model selection. The use of ONNX and Docker ensures that our best model can be deployed efficiently and reliably in any environment. The automated CI/CD pipeline further guarantees that every change is tested and deployed seamlessly, fulfilling the requirements of a modern, professional machine learning workflow.
