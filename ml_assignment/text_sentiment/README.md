# NLP: Анализ тональности русского текста

Модель: `blanchefort/rubert-base-cased-sentiment`  
Библиотека: Hugging Face `transformers` (pipeline)

## Что делает скрипт
- Берёт пример русскоязычного текста (или текст из аргумента командной строки)
- Запускает `transformers.pipeline("sentiment-analysis")`
- Печатает метку тональности и confidence

## Установка
```bash
pip install -r requirements.txt
```

## Запуск
```bash
python main.py
```

Передать свой текст:
```bash
python main.py "Мне очень понравился этот курс, всё было понятно!"
```

