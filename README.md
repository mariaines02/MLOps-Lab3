---
title: MLOps Lab2
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# MLOps Lab 2: Continuous Delivery with GitHub Actions

[![CI/CD Pipeline](https://github.com/mariaines02/MLOps-Lab2/actions/workflows/cicd.yml/badge.svg)](https://github.com/mariaines02/MLOps-Lab2/actions/workflows/cicd.yml)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project is the **continuation of MLOps Lab 1**. While Lab 1 focused on building the API and CLI, **Lab 2 focuses on automating the deployment** using Docker, GitHub Actions, Render, and Hugging Face Spaces.

**Live Demo:**
- **Frontend (Hugging Face):** [https://huggingface.co/spaces/mariaines02/mlops-lab2](https://huggingface.co/spaces/mariaines02/mlops-lab2)
- **Backend (Render):** [https://mlops-lab2-latest-7ffu.onrender.com](https://mlops-lab2-latest-7ffu.onrender.com)
- **Docker Image:** [https://hub.docker.com/repository/docker/mariaines02/mlops-lab2/general](https://hub.docker.com/repository/docker/mariaines02/mlops-lab2/general)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Lab 2 Objectives](#-lab-2-objectives)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Setup & Installation](#-setup--installation)
- [Running Locally](#-running-locally)
- [Docker Usage](#-docker-usage)
- [CI/CD Pipeline Details](#-cicd-pipeline-details)
- [Configuration](#-configuration)

---

## 🎯 Project Overview

This project implements a complete **MLOps pipeline** for an Image Classification application.
-   **Lab 1 Foundation**: Reuses the Image Classification logic (Predict, Resize, Crop, Grayscale, Normalize).
-   **Lab 2 Enhancements**: Adds containerization and automated deployment.

### Technology Stack

-   **Language:** Python 3.13
-   **API:** FastAPI + Uvicorn
-   **UI:** Gradio (hosted on Hugging Face)
-   **Container:** Docker
-   **CI/CD:** GitHub Actions
-   **Package Manager:** UV

---

## 🚀 Lab 2 Objectives

The main goal of this lab is to move from a local application to a deployed, automated system:
1.  **Containerize** the application using Docker.
2.  **Automate** testing and building using GitHub Actions.
3.  **Deploy** the API to a cloud provider (Render).
4.  **Deploy** the UI to a platform (Hugging Face Spaces).

---

## 📁 Project Structure

```
MLOps-Lab2/
├── .github/workflows/
│   └── cicd.yml          # The heart of the CI/CD pipeline
├── api/
│   └── api.py            # FastAPI backend (merged Lab 1 & 2 logic)
├── cli/
│   └── cli.py            # Command-line interface
├── logic/
│   └── predictor.py      # Image processing logic
├── templates/
│   └── home.html         # API Home page
├── tests/                # Unit and integration tests
├── app.py                # Gradio frontend application
├── Dockerfile            # Docker image definition
├── Makefile              # Development commands
├── pyproject.toml        # Project dependencies
├── requirements.txt      # Gradio dependencies
└── README.md             # Documentation
```

---

## ✨ Features

### 🧠 Image Classification API
-   **Predict**: Classify images (simulated/random for Lab 2).
-   **Image Tools**: Resize, Crop, Grayscale, Normalize.
-   **Docs**: Interactive Swagger UI at `/docs`.

### 🎨 Gradio User Interface
A user-friendly web interface hosted on Hugging Face Spaces offering tabs for:
-   🔮 **Prediction**
-   📏 **Resizing**
-   ⚫ **Grayscale Conversion**
-   ✂️ **Cropping**
-   📊 **Normalization**

---

## 🔄 CI/CD Pipeline Details

The workflow is defined in `.github/workflows/cicd.yml` and consists of two jobs:

### Job 1: `deploy-api`
1.  **Checks out the code**.
2.  **Logs in to Docker Hub** using secrets.
3.  **Builds the Docker image** using the `Dockerfile`.
4.  **Pushes the image** to Docker Hub (`mariaines02/mlops-lab2:latest`).
5.  **Triggers a redeployment on Render** using a webhook.

### Job 2: `deploy-hf` (Runs after `deploy-api`)
1.  **Checks out the `hf-space` branch**.
2.  **Pushes the contents** of this branch to your HuggingFace Space repository.

---

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.13+
-   UV (recommended) or Pip
-   Docker (optional)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mariaines02/MLOps-Lab2.git
    cd MLOps-Lab2
    ```

2.  **Install dependencies:**
    ```bash
    make install
    # OR
    uv sync
    ```

---

## 💻 Running Locally

### 1. Run the API
```bash
make run-api
```
Access at `http://localhost:8000`.

### 2. Run the Gradio App
```bash
export API_URL="http://localhost:8000"
uv run python app.py
```
Access at `http://localhost:7860`.

---

## 🐳 Docker Usage

Build and run the application in a container:

```bash
# Build
docker build -t mlops-lab2 .

# Run
docker run -p 8000:8000 mlops-lab2
```

---

## ⚙️ Configuration

To replicate this pipeline, configure these **Secrets** in GitHub:

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Docker Hub ID |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token |
| `RENDER_DEPLOY_HOOK_KEY` | Render Webhook Key |
| `HF_USERNAME` | Hugging Face Username |
| `HF_TOKEN` | Hugging Face Write Token |

**Hugging Face Configuration:**
-   Add a Variable `API_URL` in your Space settings pointing to your Render URL.

---

## 👤 Author

**Maria Ines Haddad**
-   **Course**: MLOps
-   **Lab**: 2 (Continuous Delivery)
