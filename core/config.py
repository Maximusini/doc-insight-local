import os

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_DIR = os.path.join(BASE_DIR, 'db')
BM25_PATH = os.path.join(DB_DIR, 'bm25_index.pkl')

# Модели
EMBEDDING_MODEL = 'nomic-embed-text'
LLM_MODEL = 'gemma3:4b'
RERANKER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# Параметры RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
SEARCH_TOP_K = 5     # Сколько финальных ответов отдавать
CANDIDATES_K = 10    # Сколько кандидатов искать на первом этапе