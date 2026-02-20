"""Image classification: ResNet-50 (pretrained) на CPU.

Требования:
- PyTorch + torchvision
- torchvision resnet50 pretrained=True (в новых версиях через weights=...)
- Без обучения/fine-tuning

Вход:
- example.jpg рядом со скриптом (или путь аргументом)

Выход:
- топ-5 классов ImageNet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import requests
import torch
from PIL import Image
from torchvision import models


IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def ensure_example_image(path: Path) -> Path:
    """Скачивает пример изображения, если его нет."""
    if path.exists():
        return path

    # Небольшое изображение (собака) с Wikimedia
    url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def get_imagenet_labels(cache_path: Path) -> List[str]:
    """Загружает список классов ImageNet (1000 строк). Кэшируем рядом со скриптом."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").splitlines()

    r = requests.get(IMAGENET_LABELS_URL, timeout=30)
    r.raise_for_status()
    cache_path.write_text(r.text, encoding="utf-8")
    return r.text.splitlines()


def topk(probs: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
    values, indices = torch.topk(probs, k)
    return [(int(i), float(v)) for i, v in zip(indices.cpu(), values.cpu())]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_img = base_dir / "example.jpg"

    img_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) >= 2 else default_img
    if img_path == default_img:
        img_path = ensure_example_image(img_path)

    labels = get_imagenet_labels(base_dir / "imagenet_classes.txt")

    # Pretrained ResNet-50
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()  # инференс

    # Стандартные transforms под конкретные веса
    preprocess = weights.transforms()

    image = Image.open(img_path).convert("RGB")
    x = preprocess(image).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        probs = torch.softmax(logits[0], dim=0)
        dt = time.perf_counter() - t0

    results = topk(probs, k=5)

    print("=== Image: Classification (ResNet-50, ImageNet) ===")
    print(f"Input image: {img_path}")
    print("Top-5 predictions:")
    for idx, p in results:
        name = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
        print(f"  - {name}: {p:.4f}")
    print(f"Inference time: {dt:.3f}s (CPU)")


if __name__ == "__main__":
    main()

