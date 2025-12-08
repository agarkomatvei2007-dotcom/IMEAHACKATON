from src.embedder import Embedder
from src.generator import Generator
from src.vector_store import VectorStore
from config import TOP_K, SIMILARITY_THRESHOLD
from typing import Dict, List


class RAGSystem:
    """Система RAG для ответов на вопросы (с поддержкой гибридного режима)"""
    
    def __init__(self):
        print("\n🚀 Инициализация RAG системы...")
        
        # Загружаем компоненты
        self.embedder = Embedder()
        self.generator = Generator()
        self.store = VectorStore()
        # NOTE: Предполагаем, что load_collection корректно загружает базу данных
        self.store.load_collection()
        
        print("✅ RAG система готова!\n")
    
    def ask(self, question: str, verbose: bool = False) -> Dict:
        """Задать вопрос системе"""
        
        # 1. Векторизуем вопрос
        if verbose:
            print("🔍 Ищу релевантную информацию...")
        query_embedding = self.embedder.embed(question)
        
        # 2. Ищем похожие документы (используем TOP_K из config)
        # NOTE: Предполагаем, что self.store.search принимает embedding и top_k
        results = self.store.search(query_embedding, top_k=TOP_K)
        
        # 3. Фильтруем по порогу схожести
        relevant_docs = []
        sources = []
        
        for doc, metadata, distance in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            # ChromaDB возвращает distance (расстояние), конвертируем в similarity (схожесть)
            similarity = 1 - distance 
            
            # Используем SIMILARITY_THRESHOLD из config
            if similarity >= SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)
                sources.append({
                    'text': doc,
                    'source': metadata.get('source', 'unknown'),
                    'similarity': similarity
                })
        
        if verbose:
            print(f"✅ Найдено {len(relevant_docs)} релевантных фрагментов (из {TOP_K} проверенных)\n")
        
        # 4. ФОРМИРУЕМ КОНТЕКСТ ДЛЯ ГИБРИДНОГО РЕЖИМА
        
        if relevant_docs:
            # Если нашли релевантные документы, формируем из них контекст
            context = "\n\n".join([f"Фрагмент {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)])
            
            if verbose:
                print("💬 Контекст найден. Передаю в Gemini для ответа на основе RAG.")
                
        else:
            # ЕСЛИ КОНТЕКСТ НЕ НАЙДЕН (НИЗКАЯ СХОЖЕСТЬ ИЛИ ПУСТОЙ РЕЗУЛЬТАТ)
            # Передаем пустую строку/сообщение в generator.py. 
            # Благодаря обновленному generator.py, Gemini ответит на основе общих знаний.
            context = "Контекст не найден в базе знаний."
            sources = [] # Сбрасываем источники, так как ничего релевантного не найдено
            
            if verbose:
                print("⚠️ Релевантный контекст НЕ найден. Gemini ответит на основе общих знаний.")
        
        # 5. Генерируем ответ
        if verbose:
            print("💬 Генерирую ответ...\n")
        
        # В generator.generate теперь всегда передается либо контекст, либо сообщение о его отсутствии.
        answer = self.generator.generate(question, context)
        
        return {
            'answer': answer,
            'sources': sources
        }