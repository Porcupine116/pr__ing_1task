# Audio: Speech-to-Text (ASR)

Модель: `openai/whisper-small`  
Библиотека: Hugging Face `transformers` (pipeline)

## Что делает скрипт
- Ищет файл `sample.wav` в папке рядом со скриптом
- Если файла нет — **создаёт короткий тестовый WAV** с синусоидой (это не человеческая речь, но демонстрирует запуск пайплайна)
- Прогоняет аудио через `automatic-speech-recognition` pipeline
- Печатает распознанный текст

> Для осмысленного результата положите рядом со скриптом свой файл `sample.wav` (рекомендуется 16 kHz, mono).

## Установка
```bash
pip install -r requirements.txt
```

## Запуск
```bash
python main.py
```

Передать свой путь к wav:
```bash
python main.py path\\to\\your.wav
```

