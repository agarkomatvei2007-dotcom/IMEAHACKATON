from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class Embedder:
    """Генерация эмбеддингов для текста"""
    
    def __init__(self):
        print("⚙️  Загружаю модель эмбеддингов...")
        # Используем мультиязычную модель (поддерживает русский)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Модель загружена")
    
    def embed(self, text: str) -> List[float]:
        """Создать эмбеддинг для одного текста"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Создать эмбеддинги для списка текстов"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
