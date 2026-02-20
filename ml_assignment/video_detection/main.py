"""Video (frame): Object Detection на DETR (inference-only).

Требования:
- Модель facebook/detr-resnet-50
- Hugging Face + PyTorch
- Обработать один кадр (frame.jpg)
- CPU по умолчанию

Вход:
- frame.jpg рядом со скриптом (или путь аргументом)

Выход:
- печать найденных объектов
- сохранение кадра с рамками
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor


def ensure_frame_image(path: Path) -> Path:
    """Если кадра нет — скачиваем пример и сохраняем как frame.jpg."""
    if path.exists():
        return path

    url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]


def draw_detections(image: Image.Image, detections: List[Detection]) -> Image.Image:
    """Рисуем bbox и подписи на изображении."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 3, y1 + 3), f"{det.label} {det.score:.2f}", fill="red", font=font)

    return img


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_frame = base_dir / "frame.jpg"

    frame_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) >= 2 else default_frame
    if frame_path == default_frame:
        frame_path = ensure_frame_image(frame_path)

    image = Image.open(frame_path).convert("RGB")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()

    inputs = processor(images=image, return_tensors="pt")

    t0 = time.perf_counter()
    outputs = model(**inputs)
    dt = time.perf_counter() - t0

    # Постобработка: перевод bbox к исходному размеру картинки
    # target_sizes: (height, width)
    target_sizes = [(image.height, image.width)]
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.90)[0]

    detections: List[Detection] = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = model.config.id2label[int(label_id)]
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append(
            Detection(label=label, score=float(score), box_xyxy=(x1, y1, x2, y2))
        )

    print("=== Video(frame): Object Detection (DETR) ===")
    print(f"Input frame: {frame_path}")
    print(f"Found objects: {len(detections)} (threshold=0.90)")
    for det in detections:
        print(f"  - {det.label}: {det.score:.3f} box={det.box_xyxy}")
    print(f"Inference time: {dt:.3f}s (CPU)")

    out_path = base_dir / "output_frame_annotated.jpg"
    draw_detections(image, detections).save(out_path)
    print(f"Saved annotated frame: {out_path}")


if __name__ == "__main__":
    main()

