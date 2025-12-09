import mlflow
import mlflow.pytorch
import torch
import os
from mlflow.tracking import MlflowClient


def select_and_export_model(model_name="oxford_pet_classifier"):
    client = MlflowClient()

    # Search registered models
    print(f"Searching for registered models with name '{model_name}'...")
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        print(f"No registered models found with name '{model_name}'.")
        return

    best_run_id = None
    best_val_acc = -1.0
    best_version = None

    print(f"Found {len(versions)} versions. Selecting the best one...")
    for version in versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        val_acc = run.data.metrics.get("val_acc", 0.0)
        print(f"Version {version.version} (Run {run_id}): val_acc={val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_run_id = run_id
            best_version = version

    if best_run_id is None:
        print("Could not determine best model.")
        return

    print(
        f"Best version: {best_version.version} (Run {best_run_id}) with val_acc: {best_val_acc}"
    )

    # Load best model
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Loading model from {model_uri}...")
    model = mlflow.pytorch.load_model(model_uri)
    model.to("cpu")
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    os.makedirs("results", exist_ok=True)
    onnx_path = "results/model.onnx"
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Download class labels artifact
    print("Downloading class labels...")
    try:
        local_path = client.download_artifacts(
            best_run_id, "classes.json", dst_path="results"
        )
        print(f"Downloaded classes.json to {local_path}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Failed to download classes.json: {e}")
        # Fallback: try to find it in other paths or list artifacts
        artifacts = client.list_artifacts(best_run_id)
        print("Available artifacts:")
        for art in artifacts:
            print(f" - {art.path}")

    print("Model selection and export complete.")


if __name__ == "__main__":
    select_and_export_model()
