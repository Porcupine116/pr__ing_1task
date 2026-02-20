"""Audio: Speech-to-Text (ASR) на предобученной Whisper (inference-only).

Требования:
- Модель openai/whisper-small
- Без обучения/fine-tuning
- CPU по умолчанию

Вход:
- WAV файл. По умолчанию: ./sample.wav рядом со скриптом.

Примечание:
- Если sample.wav не найден, создаётся короткий WAV с синусоидой (не речь).
  Для нормальной демонстрации распознавания речи положите свой wav.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from transformers import pipeline


def ensure_wav(path: Path, sr: int = 16000, seconds: float = 2.0) -> Tuple[Path, int]:
    """Гарантирует наличие WAV.

    Если файла нет, создаёт тестовый сигнал (синус), чтобы скрипт был полностью запускаемым.
    """
    if path.exists():
        return path, sr

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2.0 * math.pi * 440.0 * t).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, tone, sr)
    return path, sr


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_wav = base_dir / "sample.wav"

    wav_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) >= 2 else default_wav
    wav_path, _ = ensure_wav(wav_path)

    asr = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        device=-1,  # CPU
    )

    t0 = time.perf_counter()
    result = asr(str(wav_path))
    dt = time.perf_counter() - t0

    text = result.get("text") if isinstance(result, dict) else str(result)

    print("=== Audio: Speech-to-Text (Whisper) ===")
    print(f"Input wav: {wav_path}")
    print(f"Recognized text: {text}")
    print(f"Inference time: {dt:.3f}s (CPU)")


if __name__ == "__main__":
    main()

