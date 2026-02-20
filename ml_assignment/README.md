# ML Assignment — предобученные модели (inference-only)
- Для видео-детекции в задании достаточно обработать **один кадр** (`frame.jpg`). Скрипт сам создаст пример, если файл отсутствует.
- Модели при первом запуске будут скачаны из Hugging Face (нужен интернет).
- По умолчанию используется **CPU**.
## Примечания

```
  REPORT.md
    README.md
  llm_local/
    README.md
    requirements.txt
    main.py
  video_detection/
    README.md
    requirements.txt
    main.py
  image_classification/
    README.md
    requirements.txt
    main.py
  audio_speech2text/
    README.md
    requirements.txt
    main.py
  text_sentiment/
ml_assignment/
```
## Структура проекта

```
python ml_assignment/video_detection/main.py
```bash
- Video detection (по кадру `frame.jpg`):

```
python ml_assignment/image_classification/main.py
```bash
- Image classification:

```
python ml_assignment/audio_speech2text/main.py
```bash
- Audio (STT):

```
python ml_assignment/text_sentiment/main.py
```bash
- NLP (sentiment):

Каждая задача — отдельный скрипт `main.py`.
## Запуск

```
pip install -r ml_assignment/text_sentiment/requirements.txt
```bash
Пример:

- `ml_assignment/video_detection/requirements.txt`
- `ml_assignment/image_classification/requirements.txt`
- `ml_assignment/audio_speech2text/requirements.txt`
- `ml_assignment/text_sentiment/requirements.txt`
Установка зависимостей — **в каждой задаче отдельно** (так проще в университете показать независимые части):

Рекомендуется виртуальное окружение.
## Установка (Python 3.10+)

- Утилиты: `numpy`, `tqdm`
- Обработка медиа: `Pillow`, `opencv-python`, `soundfile`
- PyTorch: `torch`, `torchvision`
- Hugging Face: `transformers`, `huggingface_hub`
## Используемые библиотеки

5. **LLM (локально)** — инструкция запуска 7B модели через Ollama
4. **Video (Hugging Face + PyTorch)** — детекция объектов на **кадре** видео (объектный детектор)
3. **Image (PyTorch/Torchvision)** — классификация изображения
2. **Audio (Hugging Face Transformers)** — распознавание речи (Speech-to-Text)
1. **NLP (Hugging Face Transformers)** — анализ тональности **русского** текста (sentiment analysis)
## Задачи

Цель проекта — показать 4 базовые задачи машинного обучения **без обучения и fine-tuning**, используя только готовые предобученные модели, а также дать инструкцию по запуску локальной LLM (7–8B параметров).


