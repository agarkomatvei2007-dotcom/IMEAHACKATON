import chromadb 
from typing import List ,Dict 
from config import DB_DIR 


class VectorStore :
    """Векторное хранилище для поиска по эмбеддингам"""

    def __init__ (self ):
        self .client =chromadb .PersistentClient (path =str (DB_DIR ))
        self .collection =None 

    def create_collection (self ,name :str ="egov_docs"):
        """Создать или получить коллекцию"""
        try :
            self .client .delete_collection (name )
        except :
            pass 

        self .collection =self .client .create_collection (
        name =name ,
        metadata ={"hnsw:space":"cosine"}
        )
        print (f"✅ Коллекция '{name}' создана")

    def load_collection (self ,name :str ="egov_docs"):
        """Загрузить существующую коллекцию"""
        self .collection =self .client .get_collection (name )
        count =self .collection .count ()
        print (f"✅ Загружена коллекция: {count} документов")

    def add_documents (self ,texts :List [str ],embeddings :List [List [float ]],metadatas :List [Dict ]):
        """Добавить документы в хранилище"""
        ids =[f"doc_{i}"for i in range (len (texts ))]

        self .collection .add (
        embeddings =embeddings ,
        documents =texts ,
        metadatas =metadatas ,
        ids =ids 
        )
        print (f"✅ Добавлено {len(texts)} документов")

    def search (self ,query_embedding :List [float ],top_k :int =5 )->Dict :
        """Поиск похожих документов"""
        results =self .collection .query (
        query_embeddings =[query_embedding ],
        n_results =top_k 
        )

        return {
        'documents':results ['documents'][0 ],
        'metadatas':results ['metadatas'][0 ],
        'distances':results ['distances'][0 ]
        }
