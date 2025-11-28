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


<!-- Initial build trigger -->
This project demonstrates a complete **Continuous Integration and Continuous Delivery (CI/CD)** pipeline for a Machine Learning application. It includes a FastAPI backend, a Gradio frontend, and automated deployment using Docker Hub, Render, and HuggingFace Spaces.

## 🎯 Assignment Objective

The goal of this lab is to automate the deployment of an Image Classification application.
- **Containerization**: Use Docker to package the application.
- **CI/CD**: Use GitHub Actions to automate testing, building, and deployment.
- **Deployment**:
    - **Backend (API)**: Deployed to [Render](https://render.com) (via Docker Hub).
    - **Frontend (GUI)**: Deployed to [HuggingFace Spaces](https://huggingface.co/spaces).

## 🏗️ Project Structure

The project is organized into two main branches:
1.  **`main`**: Contains the full source code (API, Logic, CLI, Tests, Dockerfile, GitHub Actions).
2.  **`hf-space`**: A special "orphaned" branch containing *only* the files needed for the HuggingFace Space (`app.py`, `requirements.txt`, `logic/`).

## 🚀 Features

-   **Image Classification API**: A FastAPI service that predicts the class of an image (randomly for this demo).
-   **Image Processing**: Endpoints for resizing, cropping, grayscale conversion, and normalization.
-   **Gradio Interface**: A user-friendly web GUI to upload images and see predictions.
-   **Automated Pipeline**:
    -   Builds Docker image on push to `main`.
    -   Pushes image to Docker Hub.
    -   Triggers deployment on Render.
    -   Syncs the Gradio app to HuggingFace Spaces.

## 🛠️ Setup & Configuration

To make the CI/CD pipeline work, you need to configure the following **Secrets** in your GitHub Repository settings:

| Secret Name | Description |
| :--- | :--- |
| `DOCKERHUB_USERNAME` | Your Docker Hub username. |
| `DOCKERHUB_TOKEN` | Your Docker Hub Access Token (Read/Write/Delete permissions). |
| `RENDER_DEPLOY_HOOK_KEY` | The key from your Render Deploy Hook URL (the part after `?key=`). |
| `HF_USERNAME` | Your HuggingFace username. |
| `HF_TOKEN` | Your HuggingFace Access Token (Write permissions). |

## 💻 Running Locally

### 1. Run the API (Backend)
You can run the API using `uv` or `make`:
```bash
make run-api
# OR
uv run python -m api.api
```
The API will be available at `http://localhost:8000`.

### 2. Run the Gradio App (Frontend)
To run the GUI locally, you need to set the `API_URL` environment variable (optional, defaults to placeholder):
```bash
export API_URL="http://localhost:8000"
uv run python app.py
```
The Gradio app will be available at `http://localhost:7860`.

### 3. Run with Docker
Build and run the container:
```bash
docker build -t mlops-lab2 .
docker run -p 8000:8000 mlops-lab2
```

## 🔄 CI/CD Pipeline Details

The workflow is defined in `.github/workflows/cicd.yml` and consists of two jobs:

### Job 1: `deploy-api`
1.  Checks out the code.
2.  Logs in to Docker Hub.
3.  Builds the Docker image using the `Dockerfile`.
4.  Pushes the image to Docker Hub.
5.  Triggers a redeployment on Render using a webhook.

### Job 2: `deploy-hf` (Runs after `deploy-api`)
1.  Checks out the `hf-space` branch.
2.  Pushes the contents of this branch to your HuggingFace Space repository.

---
**Author**: Maria Ines Haddad
**Course**: MLOps
