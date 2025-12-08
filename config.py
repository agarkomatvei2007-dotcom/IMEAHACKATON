import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_DIR = DATA_DIR / "vector_db"

# API ключ для генерации ответов
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Параметры RAG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5
SIMILARITY_THRESHOLD = 0.65

# Создаём директории если их нет
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
