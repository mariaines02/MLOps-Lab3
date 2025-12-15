#set document(title: "MLOps Lab 3 Report", author: "Maria Ines")

// --- Configuration & Styling ---
#let accent-color = rgb("#2A9D8F") // Teal
#let dark-color = rgb("#264653")   // Dark Blue-Grey

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(style: "italic", fill: gray)
      #align(right)[MLOps Laboratory Series | Final Report]
      #line(length: 100%, stroke: 0.5pt + gray)
    ]
  }
)

#set text(
  font: "Linux Biolinum", // Sans-serif alternative if available, or fallback
  size: 11pt,
  fill: rgb("#333333")
)

#set par(
  justify: true, 
  leading: 0.8em,
  first-line-indent: 0em
)

// Custom Heading Styling
#show heading: it => [
  #set text(fill: dark-color, weight: "bold")
  #v(0.5em)
  #if it.level == 1 [
    #set text(size: 18pt)
    #block(stroke: (bottom: 2pt + accent-color), inset: (bottom: 5pt), width: 100%)[
      #upper(it.body)
    ]
  ] else [
    #set text(size: 14pt)
    #it.body
  ]
  #v(0.5em)
]


// --- Title Page ---
#align(center + horizon)[
  #box(
    width: 100%,
    stroke: 2pt + accent-color,
    inset: 40pt,
    radius: 10pt,
    fill: rgb("#f0fdfa") // Very light teal background
  )[
    #text(size: 32pt, weight: "black", fill: dark-color)[MLOps]\
    #v(0.5em)
    #text(size: 20pt, weight: "regular")[From Logic to Production]
    
    #line(length: 50%, stroke: 2pt + accent-color)
    
    #v(1cm)
    #text(size: 14pt, style: "italic")[
      Final Laboratory Report\
      CI/CD • Containerization • MLFlow
    ]
    
    #v(2cm)
    #text(size: 12pt, weight: "bold")[Maria Ines] \
    #text(size: 12pt)[Machine Learning Operations] \
    #text(size: 12pt)[#datetime.today().display("[month repr:long] [day], [year]")]
  ]
]

#pagebreak()

// --- Content ---

= Executive Summary

This report details the implementation of an end-to-end MLOps pipeline for the **Pet Classification System**. Moving beyond simple scripts to workflow integrating:

- **Continuous Integration (GitHub Actions)** for code quality.
- **Continuous Delivery (Docker & Render)** for automated deployment.
- **Experiment Tracking (MLFlow)** for data-driven model selection.

The final system leverages a **MobileNet_v2** architecture, optimized with **ONNX**, to classify 37 pet breeds with high accuracy and efficiency.

= Project Resources

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 15pt,
  rect(width: 100%, stroke: 1pt + accent-color, radius: 5pt, inset: 12pt, fill: rgb("#f0fdfa"))[
    #align(center)[*Lab 1*]
    #v(0.5em)
    #list(marker: box(fill: accent-color, width: 3pt, height: 3pt, radius: 1.5pt),
      link("https://github.com/mariaines02/MLOps-Lab1")[GitHub Repository]
    )
  ],
  rect(width: 100%, stroke: 1pt + accent-color, radius: 5pt, inset: 12pt, fill: rgb("#f0fdfa"))[
    #align(center)[*Lab 2*]
    #v(0.5em)
    #list(marker: box(fill: accent-color, width: 3pt, height: 3pt, radius: 1.5pt),
      link("https://github.com/mariaines02/MLOps-Lab2")[GitHub Repository],
      link("https://huggingface.co/spaces/mariaines02/mlops-lab2")[Hugging Face Space],
      link("https://mlops-lab2-latest-7ffu.onrender.com")[Render Deployment]
    )
  ],
  rect(width: 100%, stroke: 1pt + accent-color, radius: 5pt, inset: 12pt, fill: rgb("#f0fdfa"))[
    #align(center)[*Lab 3*]
    #v(0.5em)
    #list(marker: box(fill: accent-color, width: 3pt, height: 3pt, radius: 1.5pt),
      link("https://github.com/mariaines02/MLOps-Lab3")[GitHub Repository],
      link("https://huggingface.co/spaces/mariaines02/mlops-lab3")[Hugging Face Space],
      link("https://dashboard.render.com/web/srv-d4rv368gjchc7382vbqg/deploys/dep-d5028t7gi27c73deo1jg")[Render Dashboard]
    )
  ]
)

= Lab 1: Continuous Integration using GitHub Actions

The first phase focused on establishing a robust software engineering foundation for Machine Learning. I implemented a CI pipeline using **GitHub Actions**.

== Core Architecture
- **Logic (`logic/`)**: Encapsulates core functionality like image prediction and preprocessing using **Pillow (PIL)**.
- **API (`api/`)**: A **FastAPI** application exposing the logic via REST endpoints.
- **CLI (`cli/`)**: A command-line interface using **Click** for local execution and debugging.

== Continuous Integration Pipeline
The CI pipeline (`ci.yml`) ensures code quality on every push to `main` or `develop`.
1.  **Setup**: Installs Python 3.11 and the **UV** package manager.
2.  **Formatting**: Verifies code style with **Black**.
3.  **Linting**: Checks code quality with **Pylint**.
4.  **Testing**: Executes unit and integration tests with **Pytest** and **Pytest-Cov**.

= Lab 2: Continuous Delivery using GitHub
Actions

The second phase focused on automating deployment and ensuring consistency across environments using **Docker**.

== Containerization
I created a multi-stage `Dockerfile` to optimize the image size:
- **Base**: Python 3.13 slim image.
- **Builder**: Installs dependencies and compiles packages.
- **Runtime**: Copies only necessary artifacts, resulting in a lightweight production image.

== Deployment Architecture
I adopted a decoupled architecture for scalability:
1.  **Backend (Render)**: The FastAPI application is containerized and deployed to Render. It handles the heavy lifting of inference and image processing.
2.  **Frontend (Hugging Face)**: A user-friendly **Gradio** interface is deployed to Hugging Face Spaces. It communicates with the backend via HTTP requests.

== CD Pipeline
The CD workflow (`cicd.yml`) consists of two jobs:
1.  **`deploy-api`**: Builds the Docker image, pushes it to **Docker Hub**, and triggers a redeployment on Render via webhook.
2.  **`deploy-hf`**: Pushes the frontend code to the Hugging Face Space repository.

= Lab 3: Experiment tracking and versioning with MLFlow

The final phase replaced the random predictor with a real Deep Learning model, introducing rigorous MLOps practices.

== Transfer Learning
I employed **MobileNet_v2**, pre-trained on ImageNet, and fine-tuned it for the **Oxford-IIIT Pet Dataset** (37 classes). The feature extractor was frozen, and only the classification head was trained.

== Experiment Tracking Strategy
I utilized **MLFlow** to ensure full reproducibility and traceability of the machine learning lifecycle. For each training run, I systematically logged:

- **Hyperparameters**: Key configuration parameters such as `batch_size`, `learning_rate`, `optimizer` (Adam), and `seed` (42) were recorded to ensure fair comparisons between experiments.
- **Metrics**: `train_loss`, `val_loss`, `train_acc`, and `val_acc` were logged at every epoch. This allowed me to visualize learning curves in the MLFlow UI and detect issues like overfitting or poor convergence early.
- **Artifacts**: The `classes.json` file and the final model weights were stored as artifacts, linking specific code versions to their outputs.

== Analysis & Model Selection
Using the **MLFlow UI Dashboard**, I performed a comparative analysis of four distinct experimental runs, testing both hyperparameter variations and different model architectures.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 8pt,
    align: center,
    stroke: none,
    fill: (x, y) => if y == 0 { accent-color.lighten(80%) } else if x == 0 { gray.lighten(90%) },
    
    [*Run Name*], [*Model*], [*Batch*], [*LR*], [*Val Acc*],
    [mobilenet_bs32_lr0.01], [MobileNet], [32], [0.01], [61%],
    [mobilenet_bs32_lr0.001], [MobileNet], [32], [0.001], [77%],
    [resnet18_bs32_lr0.001], [ResNet18], [32], [0.001], [76%],
    [*mobilenet_bs4_lr0.001*], [*MobileNet*], [*4*], [*0.001*], [*79.99%*],
  ),
  caption: "Performance Comparison from MLFlow"
)

**Key Observations:**
1.  **Architecture Comparison**: I tested **ResNet18** against **MobileNet_v2**. While ResNet18 is a powerful model, MobileNet_v2 achieved slightly better accuracy and is significantly more lightweight, making it ideal for our resource-constrained deployment.
2.  **Hyperparameter Tuning**: 
    - High learning rates (0.01) caused instability.
    - Reducing the batch size to 4 for MobileNet provided the best generalization, likely due to the regularization effect of noisier gradient updates.

Based on this comprehensive analysis, I selected the **MobileNet_v2 (Run 4)** as the production model. It offers the best balance of high accuracy and efficiency.

= 4. User Interfaces

This section showcases the various interfaces available for interacting with the system, catering to different user needs (developer, data scientist, end-user).

== REST API Interface
The **FastAPI** documentation (Swagger UI) provides a interface to test endpoints and understand the API schema.

#figure(
  image("apirest1.png", width: 90%),
  caption: "FastAPI Documentation - Endpoints"
)

#figure(
  image("apirest2.png", width: 90%),
  caption: "FastAPI Prediction Endpoint"
)

== MLFlow Interface
The **MLFlow UI** serves as the central hub for experiment tracking, allowing for detailed comparison of metrics and parameters across different runs.

#figure(
  image("mlflow ui.png", width: 90%),
  caption: "MLFlow Experiment Dashboard"
)

== Hugging Face Interface
The frontend application provides an interactive interface for users to test the model and image processing tools.

=== Prediction Interface
The main interface allows users to upload an image and receive a breed classification.

#figure(
  image("predict.png", width: 80%),
  caption: "Gradio Prediction Interface"
)

== Image Preprocessing Tools
The application also exposes the backend's image processing capabilities.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Resize
    #figure(
      image("resize.png", width: 100%),
      caption: "Resize Tool"
    )
  ],
  [
    === Normalize
    #figure(
      image("normalize.png", width: 100%),
      caption: "Normalization Tool"
    )
  ]
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Grayscale
    #figure(
      image("grayscale.png", width: 100%),
      caption: "Grayscale Conversion"
    )
  ],
  [
    === Crop
    #figure(
      image("crop.png", width: 100%),
      caption: "Crop Tool"
    )
  ]
)

= Conclusion

I have successfully transformed a basic concept into a production-grade ML system. The integration of **MLFlow** for tracking, **ONNX** for optimization, and **Docker** for deployment demonstrates a complete, professional MLOps lifecycle.
