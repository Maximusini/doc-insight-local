import chromadb
import ollama
import pickle
import os
import rank_bm25
from core.utils import tokenize, generate_chunk_id
from sentence_transformers import CrossEncoder
from core.config import *

class RAGClient:
    def __init__(self, db_path=DB_DIR, collection_name='docs', llm_model=LLM_MODEL):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.llm_model = llm_model
        
        try:
            available_models = [m['model'] for m in ollama.list()['models']]
            base_model = EMBEDDING_MODEL.split(':')[0]
            
            if base_model not in available_models:
                ollama.pull(EMBEDDING_MODEL)
        except Exception as e:
            print(e)
        
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, 'rb') as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = None
        
        try:
            self.bm25_chunks = self.collection.get()['documents']
        except Exception:
            self.bm25_chunks = []
            
    def get_embedding(self, text):
        text_emb = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return text_emb['embedding']
    
    def add_documents(self, documents):
        ids = [generate_chunk_id(doc) for doc in documents]
        embeddings = [self.get_embedding(doc) for doc in documents]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )
            
    def build_indices(self, new_chunks):
        self.add_documents(new_chunks)
        
        all_docs = self.collection.get()['documents']
        
        tokenized_corpus = [tokenize(doc) for doc in all_docs]
        bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        with open(BM25_PATH, 'wb') as f: 
            pickle.dump(bm25, f)
            
        self.bm25 = bm25
        self.bm25_chunks = all_docs  
            
    def query_bm25(self, text, n=SEARCH_TOP_K):
        tokenized_text = tokenize(text)
        
        if self.bm25 != None:
            top = self.bm25.get_top_n(tokenized_text, self.bm25_chunks, n=n)
            return top
        else:
            return []
            
    def query(self, text, n_results=SEARCH_TOP_K):
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
    
    def contextualize_query(self, query, chat_history):
        last_5 = chat_history[-5:]
        last_messages = []
        for item in last_5:
            role = item['role']
            content = item['content']
            message = f'{role}: {content}'
            last_messages.append(message)
        last_messages_str = '\n'.join(last_messages)
        instruction = f'''System prompt. Ты переписываешь вопросы для поисковой системы. 
                          Учитывая историю чата, переформулируй последний вопрос пользователя так, чтобы он стал самостоятельным и полным.
                          Верни ТОЛЬКО переформулированный вопрос.
                          User prompt. История: {last_messages_str}, Вопрос: {query}.
                          '''
        response = ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': instruction}])
        
        return response['message']['content']
        
    def generate_answer(self, context, question):
        if not context:
            return 'Я не нашёл информацию в документах.'

        instruction = f'''Используя только следующий контекст, ответь на вопрос на русском языке.
                          Контекст: {context}
                          Вопрос: {question}'''
                          
        response = ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': instruction}])
        
        return response['message']['content']
    
    def reset_database(self):
        try:
            self.client.delete_collection(self.collection.name)
        except ValueError:
            pass
        
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
        
        self.bm25 = None
        self.bm25_chunks = []