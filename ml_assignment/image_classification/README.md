# Image: Классификация изображения (ResNet-50)

Модель: `torchvision.models.resnet50` (предобученные веса)  
Фреймворк: PyTorch/Torchvision

## Что делает скрипт
- Берёт изображение `example.jpg` рядом со скриптом
- Если файла нет — скачивает небольшой пример изображения
- Запускает ResNet-50 и выводит **топ-5** классов ImageNet

## Установка
```bash
py -m pip install -r requirements.txt
```

## Запуск
```bash
py main.py
```

Передать свой путь к изображению:
```bash
py main.py path\\to\\image.jpg
```

