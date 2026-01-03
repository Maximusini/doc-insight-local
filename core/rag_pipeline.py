import chromadb
import ollama
import numpy as np
import re
import pickle
from core.utils import tokenize
from sentence_transformers import CrossEncoder

class RAGClient:
    def __init__(self, db_path='./db', collection_name='docs'):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        self.reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        
        with open('bm25_index.pkl', 'rb') as f:
            bm25_index = pickle.load(f)
        self.bm25 = bm25_index['model']
        self.bm25_chunks = bm25_index['chunks']
        
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
            
    def query_bm25(self, text, n=3):
        tokenized_text = tokenize(text)
        top = self.bm25.get_top_n(tokenized_text, self.bm25_chunks, n=n)
        return top
            
    def query(self, text, n_results=3):
        n_candidates = n_results * 3
        vec = self.get_embedding(text)
        qbm = self.query_bm25(text, n=n_candidates)
        results = self.collection.query(
            query_embeddings=[vec],
            n_results=n_candidates
        )
        
        if results['documents'] and results['documents'][0]:
            vector_docs = results['documents'][0]
        else:
            vector_docs = []
            
        combined = list(set(vector_docs + qbm))
        
        pairs = [[text, candidate] for candidate in combined]
        scores = self.reranker.predict(pairs)
        candidates = [candidate[1] for candidate in pairs]
        
        candidate_scores = list(zip(candidates, scores))
        sorted_scores =  sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        top_n = sorted_scores[:n_results]
        final_docs = [doc[0] for doc in top_n]
        result = '\n---\n'.join(final_docs)
        
        return result
        
    def generate_answer(self, context, question):
        if not context:
            return 'Я не нашёл информацию в документах.'

        instruction = f'''Используя только следующий контекст, ответь на вопрос на русском языке.
                          Контекст: {context}
                          Вопрос: {question}'''
                          
        response = ollama.chat(model='gemma3:4b', messages=[{'role': 'user', 'content': instruction}])
        
        return response['message']['content']