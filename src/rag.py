from src .embedder import Embedder 
from src .generator import Generator 
from src .vector_store import VectorStore 
from config import TOP_K ,SIMILARITY_THRESHOLD 
from typing import Dict ,List 


class RAGSystem :
    """Система RAG для ответов на вопросы (с поддержкой гибридного режима)"""

    def __init__ (self ):
        print ("\n🚀 Инициализация RAG системы...")


        self .embedder =Embedder ()
        self .generator =Generator ()
        self .store =VectorStore ()

        self .store .load_collection ()

        print ("✅ RAG система готова!\n")

    def ask (self ,question :str ,verbose :bool =False )->Dict :
        """Задать вопрос системе"""


        if verbose :
            print ("🔍 Ищу релевантную информацию...")
        query_embedding =self .embedder .embed (question )



        results =self .store .search (query_embedding ,top_k =TOP_K )


        relevant_docs =[]
        sources =[]

        for doc ,metadata ,distance in zip (
        results ['documents'],
        results ['metadatas'],
        results ['distances']
        ):

            similarity =1 -distance 


            if similarity >=SIMILARITY_THRESHOLD :
                relevant_docs .append (doc )
                sources .append ({
                'text':doc ,
                'source':metadata .get ('source','unknown'),
                'similarity':similarity 
                })

        if verbose :
            print (f"✅ Найдено {len(relevant_docs)} релевантных фрагментов (из {TOP_K} проверенных)\n")



        if relevant_docs :

            context ="\n\n".join ([f"Фрагмент {i+1}:\n{doc}"for i ,doc in enumerate (relevant_docs )])

            if verbose :
                print ("💬 Контекст найден. Передаю в Gemini для ответа на основе RAG.")

        else :



            context ="Контекст не найден в базе знаний."
            sources =[]

            if verbose :
                print ("⚠️ Релевантный контекст НЕ найден. Gemini ответит на основе общих знаний.")


        if verbose :
            print ("💬 Генерирую ответ...\n")


        answer =self .generator .generate (question ,context )

        return {
        'answer':answer ,
        'sources':sources 
        }