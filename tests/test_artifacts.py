import os
import pytest

def test_model_artifacts_exist():
    assert os.path.exists("results/model.onnx"), "model.onnx not found"
    assert os.path.exists("results/classes.json"), "classes.json not found"

if __name__ == "__main__":
    test_model_artifacts_exist()
    print("Artifacts existence test passed.")
