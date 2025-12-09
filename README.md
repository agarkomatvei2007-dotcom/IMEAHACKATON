### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка API ключа

Получите API ключ: https://makersuite.google.com/app/apikey

Создайте файл `.env`:
```bash
cp .env.example .env
```

Откройте `.env` и укажите ваш ключ:
```
GEMINI_API_KEY=ваш_ключ_здесь
```

### 3. Запуск приложения

```bash
streamlit run app.py
```

