import chromadb
import ollama
import numpy as np

class RAGClient:
    def __init__(self, db_path='./db', collection_name='docs'):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def get_embedding(self, text):
        text_emb = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return text_emb['embedding']
    
    def add_documents(self, documents):
        for i, doc in enumerate(documents):
            vec = self.get_embedding(doc)
            self.collection.add(
                ids=[f'doc_{len(self.collection.get()["ids"]) + i}'],
                embeddings=[vec],
                documents=[doc]
            )
            
    def query(self, text, n_results=1):
        vec = self.get_embedding(text)
        results = self.collection.query(
            query_embeddings=[vec],
            n_results=n_results
        )
        if results['documents'] and results['documents'][0]:
            return results['documents'][0][0]
        return None
    
    def generate_answer(self, context, question):
        if not context:
            return 'Я не нашёл информацию в документах.'

        instruction = f'''Используя только следующий контекст, ответь на вопрос на русском языке.
                          Контекст: {context}
                          Вопрос: {question}'''
                          
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': instruction}])
        
        return response['message']['content']