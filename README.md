---
title: MLOps Lab 3 - Pet Classifier
emoji: 🐾
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: mit
---

# MLOps Lab 3: Experiment Tracking & Model Deployment

[![CI/CD Pipeline](https://github.com/mariaines02/MLOps-Lab3/actions/workflows/cicd.yml/badge.svg)](https://github.com/mariaines02/MLOps-Lab3/actions/workflows/cicd.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![MLFlow](https://img.shields.io/badge/MLFlow-Tracking-blue.svg)](https://mlflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-grey.svg)](https://onnx.ai/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

This project is the **final stage** of the MLOps Laboratory series. It builds upon the foundations of Lab 1 (API/CLI) and Lab 2 (Docker/CD) to introduce **Experiment Tracking** and **Model Versioning**.

**Live Demo:**
- **Frontend (Hugging Face):** [https://huggingface.co/spaces/mariaines02/mlops-lab3](https://huggingface.co/spaces/mariaines02/mlops-lab3)
- **Backend (Render):** [https://mlops-lab3-latest.onrender.com](https://mlops-lab3-latest.onrender.com)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Lab 3 Objectives](#-lab-3-objectives)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Setup & Installation](#-setup--installation)
- [Workflow](#-workflow)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)

---

## 📚 Previous Labs

This project is part of a comprehensive MLOps series:

| Lab | Focus | Resources |
| :--- | :--- | :--- |
| **Lab 1** | **Logic & CI** | [GitHub](https://github.com/mariaines02/MLOps-Lab1) |
| **Lab 2** | **Docker & CD** | [GitHub](https://github.com/mariaines02/MLOps-Lab2) • [Hugging Face](https://huggingface.co/spaces/mariaines02/mlops-lab2) • [Render](https://mlops-lab2-latest-7ffu.onrender.com) |
| **Lab 3** | **Tracking & ML** | [GitHub](https://github.com/mariaines02/MLOps-Lab3) • [Hugging Face](https://huggingface.co/spaces/mariaines02/mlops-lab3) • [Render](https://mlops-lab3-latest.onrender.com) |

---

## 🎯 Project Overview

In this lab, we replace the random prediction logic with a real **Deep Learning model** trained on the **Oxford-IIIT Pet Dataset**.
-   **Models Experimented**: MobileNet_v2, ResNet18 (Transfer Learning)
-   **Tracking**: MLFlow for experiments, metrics, and model registry
-   **Inference**: ONNX Runtime for efficient production deployment

### Technology Stack

-   **Training**: PyTorch + MobileNet_v2 / ResNet18
-   **Tracking**: MLFlow
-   **Serialization**: ONNX
-   **API**: FastAPI
-   **Frontend**: Gradio
-   **Container**: Docker

---

## 🚀 Lab 3 Objectives

1.  **Train** deep learning models (MobileNet, ResNet) to classify 37 pet breeds.
2.  **Track** experiments (metrics, params, models) using MLFlow.
3.  **Compare** models in the MLFlow UI and register the best one.
4.  **Deploy** the optimized ONNX model to production.

---

## 📁 Project Structure

```
MLOps-Lab3/
├── src/
│   ├── train.py          # Training script (supports --model_name)
│   ├── select_model.py   # Model selection & ONNX export
│   └── inference.py      # ONNX inference demo
├── logic/
│   └── predictor.py      # Production inference logic
├── api/
│   └── api.py            # FastAPI application
├── .github/workflows/
│   └── cicd.yml          # CI/CD Pipeline
├── Dockerfile            # Multi-stage Docker build
├── app.py                # Gradio Frontend
└── requirements.txt      # Project dependencies
```

---

## ✨ Features

### 🧠 Intelligent Prediction
-   Classifies images into **37 different pet breeds**.
-   **Model Selection**: Best performing model selected from multiple architectures (MobileNet vs ResNet).
-   **Confidence Scores** returned for each prediction.

### 📊 Experiment Tracking (MLFlow)
-   Logs **Hyperparameters** (learning rate, batch size, model type).
-   Logs **Metrics** (accuracy, loss).
-   **Model Registry**: Versioning of candidate and production models.

### ⚡ Optimized Inference
-   Models are converted to **ONNX** format.
-   **Quantization-ready** and platform-independent.
-   No heavy PyTorch dependencies needed in production inference.

---

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.11+
-   UV Package Manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mariaines02/MLOps-Lab3.git
    cd MLOps-Lab3
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

3.  **Activate environment:**
    ```bash
    source .venv/bin/activate
    ```

---

## 🔄 Workflow

### 1. Train Models
Run the training script to fine-tune different models.
```bash
# Train MobileNet
uv run python src/train.py --model_name mobilenet_v2 --epochs 3

# Train ResNet18
uv run python src/train.py --model_name resnet18 --epochs 3
```

### 2. Compare Experiments
Launch the MLFlow UI to compare runs and models.
```bash
uv run mlflow ui
```
Visit `http://127.0.0.1:5000`. Go to the **Models** tab to see registered candidates.

### 3. Select Best Model
Automatically pick the best run and export to ONNX.
```bash
uv run python src/select_model.py
```
This creates `results/model.onnx` and `results/classes.json`.

### 4. Run API Locally
```bash
uv run uvicorn api.api:app --reload
```

---

## 📈 Experiment Tracking Strategy

We use **MLFlow** to track every aspect of the training process:

1.  **Hyperparameters**: We log `batch_size`, `learning_rate`, `epochs`, and `model_name` for every run.
2.  **Metrics**: `train_loss`, `val_loss`, `train_acc`, and `val_acc` are logged at each epoch to visualize learning curves.
3.  **Artifacts**:
    -   `classes.json`: Ensures we know the class mapping.
    -   `model`: The full PyTorch model.
    -   `training_curves.png`: Visual plots of the training progress.
4.  **Model Registry**:
    -   **Candidates**: All runs are registered as candidates.
    -   **Production**: The best performing model (highest validation accuracy) is promoted to production.

---

## ☁️ Deployment

### Backend (Render)
The API is containerized and deployed to Render.
-   **Docker Image**: Built from `Dockerfile`.
-   **Platform**: Render Web Service.

### Frontend (Hugging Face)
The Gradio UI connects to the backend API.
-   **Space**: [mariaines02/mlops-lab3](https://huggingface.co/spaces/mariaines02/mlops-lab3)
-   **Configuration**: `API_URL` secret points to the Render backend.

---

## 🔄 CI/CD Pipeline

The project uses **GitHub Actions** for automation:

1.  **CI (Continuous Integration)**:
    -   Lints code with `pylint` and `black`.
    -   Runs tests with `pytest`.

2.  **CD (Continuous Delivery)**:
    -   Builds the Docker image.
    -   Pushes to **Docker Hub**.
    -   Triggers **Render** deployment.
    -   Updates **Hugging Face Space**.

---

## 👤 Author

**Maria Ines Haddad**
-   **Course**: MLOps
-   **Lab**: 3 (Final Project)
