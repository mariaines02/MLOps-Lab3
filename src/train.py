import argparse
import json
import os
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


def train_model(params):
    # Set seeds for reproducibility
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # Define transforms
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    print("Loading dataset...")
    # We use the OxfordIIITPet dataset.
    # Note: The dataset might need to be downloaded.
    # We will store it in a 'data' directory.
    os.makedirs("data", exist_ok=True)

    try:
        full_dataset = datasets.OxfordIIITPet(
            root="data", split="trainval", download=True, transform=data_transforms
        )
    except RuntimeError:
        print(
            "Dataset might already exist or download failed. Trying to load without download."
        )
        full_dataset = datasets.OxfordIIITPet(
            root="data", split="trainval", download=False, transform=data_transforms
        )

    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=2
    )

    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Save class labels
    os.makedirs("results", exist_ok=True)
    with open("results/classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f)

    # Load model
    print(f"Loading model {params.model_name}...")
    if params.model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        # Freeze parameters
        for param in model.features.parameters():
            param.requires_grad = False

        # Modify classifier
        # MobileNetV2 classifier is a Sequential with a Dropout and a Linear layer.
        # We replace the last Linear layer.
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        raise ValueError(f"Model {params.model_name} not supported yet.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=params.learning_rate)

    # MLFlow setup
    mlflow.set_experiment(params.experiment_name)

    run_name = f"{params.model_name}_bs{params.batch_size}_lr{params.learning_rate}"

    print(f"Starting run: {run_name}")
    with mlflow.start_run(run_name=run_name) as _:
        # Log params
        mlflow.log_params(
            {
                "model_name": params.model_name,
                "batch_size": params.batch_size,
                "learning_rate": params.learning_rate,
                "epochs": params.epochs,
                "seed": params.seed,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
            }
        )

        # Log artifacts (class labels)
        mlflow.log_artifact("results/classes.json")

        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for epoch in range(params.epochs):
            print(f"Epoch {epoch+1}/{params.epochs}")

            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_size
            epoch_acc = running_corrects.double() / train_size

            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc.item(), step=epoch)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            val_loss = val_loss / val_size
            val_acc = val_corrects.double() / val_size

            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc.item(), step=epoch)

            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc.item())
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc.item())

        # Plot and log curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Val Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label="Train Acc")
        plt.plot(val_acc_history, label="Val Acc")
        plt.legend()
        plt.title("Accuracy")

        os.makedirs("plots", exist_ok=True)
        plot_path = "plots/training_curves.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        # Log model
        print("Logging model...")

        # Infer signature
        from mlflow.models.signature import infer_signature

        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = model(dummy_input)
        signature = infer_signature(
            dummy_input.cpu().numpy(), dummy_output.cpu().detach().numpy()
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name="oxford_pet_classifier",
            signature=signature,
            input_example=dummy_input.cpu().numpy(),
        )

        print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on Oxford-IIIT Pet dataset"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_name", type=str, default="mobilenet_v2", help="Model name"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="OxfordPetExperiment",
        help="MLFlow experiment name",
    )

    args = parser.parse_args()
    train_model(args)
