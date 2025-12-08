"""
Скрипт для обработки документов и создания векторной базы
"""
from pathlib import Path
from src.embedder import Embedder
from src.vector_store import VectorStore
from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
import re


def load_document(filepath: Path) -> str:
    """Загрузить документ из файла"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Разбить текст на чанки"""
    # Разбиваем по параграфам
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Если параграф слишком большой, разбиваем по предложениям
        if len(para) > chunk_size:
            sentences = re.split(r'[.!?]\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
        else:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def main():
    print("🚀 Создание векторной базы знаний\n")
    
    # Загружаем все документы
    print("📂 Загрузка документов...")
    docs = list(DOCS_DIR.glob("*.md"))
    
    if not docs:
        print("❌ Не найдено документов в", DOCS_DIR)
        return
    
    print(f"✅ Найдено {len(docs)} документов\n")
    
    all_chunks = []
    all_metadatas = []
    
    for doc in docs:
        print(f"📄 Обработка: {doc.name}")
        text = load_document(doc)
        chunks = chunk_text(text)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                'source': doc.name,
                'filename': doc.stem
            })
        
        print(f"   └─ Создано {len(chunks)} чанков")
    
    print(f"\n📊 Всего чанков: {len(all_chunks)}\n")
    
    # Создаём эмбеддинги
    print("🔄 Создание эмбеддингов...")
    embedder = Embedder()
    embeddings = embedder.embed_batch(all_chunks)
    print(f"✅ Создано {len(embeddings)} эмбеддингов\n")
    
    # Сохраняем в векторную базу
    print("💾 Сохранение в векторную базу...")
    store = VectorStore()
    store.create_collection()
    store.add_documents(all_chunks, embeddings, all_metadatas)
    
    print("\n✅ Векторная база готова!")
    print("Теперь запусти: python app.py\n")


if __name__ == "__main__":
    main()
