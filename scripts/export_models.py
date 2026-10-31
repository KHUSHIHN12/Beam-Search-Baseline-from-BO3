"""export_models.py - Exports PyTorch models to ONNX and imports into IREE MLIR.

Follows the paper's specification:
- PyTorch + ONNX opset 18
- iree-import-onnx (python -m iree.compiler.tools.import_onnx)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import torch
import torch.nn as nn
from typing import Tuple


class SmallCNN(nn.Module):
    """A compact 3-layer CNN for ultra-fast compilation (<5s) during live demos."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SimpleMLP(nn.Module):
    """Dense network with NO convolutions - used to test legality masking."""

    def __init__(self, in_features: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model_and_dummy(model_name: str) -> Tuple[nn.Module, torch.Tensor]:
    """Instantiates the PyTorch model and an appropriately-shaped dummy input."""
    name = model_name.lower().replace("-", "_")
    if name == "small_cnn":
        model = SmallCNN()
        dummy = torch.randn(1, 3, 32, 32)
    elif name == "mlp":
        model = SimpleMLP()
        dummy = torch.randn(1, 128)
    elif name == "resnet18":
        import torchvision.models as models

        model = models.resnet18(weights=None)
        dummy = torch.randn(1, 3, 224, 224)
    elif name == "mobilenet_v2":
        import torchvision.models as models

        model = models.mobilenet_v2(weights=None)
        dummy = torch.randn(1, 3, 224, 224)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: small_cnn, mlp, resnet18, mobilenet_v2"
        )

    model.eval()
    return model, dummy


def export_model_to_onnx(model_name: str, output_dir: str) -> str:
    """Exports model to ONNX format with opset 18."""
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"[*] Exporting PyTorch model '{model_name}' -> {onnx_path}...")
    model, dummy = get_model_and_dummy(model_name)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    [OK] ONNX model exported ({size_mb:.2f} MB)")
    return onnx_path


def import_onnx_to_mlir(onnx_path: str, mlir_path: Optional[str] = None) -> Optional[str]:
    """Converts ONNX model to MLIR using iree.compiler.tools.import_onnx."""
    if mlir_path is None:
        mlir_path = os.path.splitext(onnx_path)[0] + ".mlir"

    print(f"[*] Importing ONNX -> MLIR ({mlir_path})...")
    cmd = [
        sys.executable,
        "-m",
        "iree.compiler.tools.import_onnx",
        onnx_path,
        "-o",
        mlir_path,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and os.path.exists(mlir_path) and os.path.getsize(mlir_path) > 0:
            size_mb = os.path.getsize(mlir_path) / (1024 * 1024)
            print(f"    [OK] IREE MLIR imported ({size_mb:.2f} MB)")
            return mlir_path
        else:
            print(f"    [!] Failed to import MLIR: {res.stderr[:200]}")
            return None
    except Exception as e:
        print(f"    [!] Exception running iree-import-onnx: {e}")
        return None


def export_and_import(model_name: str, output_dir: str = "models") -> Tuple[str, Optional[str]]:
    """Complete export pipeline: PyTorch -> ONNX -> MLIR."""
    onnx_file = export_model_to_onnx(model_name, output_dir)
    mlir_file = import_onnx_to_mlir(onnx_file)
    return onnx_file, mlir_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX and IREE MLIR")
    parser.add_argument(
        "--model",
        type=str,
        default="small_cnn",
        choices=["small_cnn", "mlp", "resnet18", "mobilenet_v2", "all"],
        help="Model to export",
    )
    parser.add_argument("--output-dir", type=str, default="models", help="Destination folder")
    args = parser.parse_args()

    targets = ["small_cnn", "mlp", "resnet18", "mobilenet_v2"] if args.model == "all" else [args.model]
    for target in targets:
        export_and_import(target, args.output_dir)

