# Video: Детекция объектов на кадре (DETR)

Модель: `facebook/detr-resnet-50`  
Библиотеки: Hugging Face `transformers` + PyTorch

## Что делает скрипт
- Берёт кадр `frame.jpg` рядом со скриптом
- Если файла нет — скачивает пример изображения и использует его как «кадр видео»
- Запускает DETR и находит объекты (bbox + класс)
- Сохраняет изображение с рамками в `output_frame_annotated.jpg`

## Установка
```bash
py -m pip install -r requirements.txt
```

## Запуск
```bash
py main.py
```

Передать свой кадр:
```bash
py main.py path\\to\\frame.jpg
```

