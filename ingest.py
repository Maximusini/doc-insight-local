import rank_bm25
import pickle
import re
from core.reader import read_pdf
from core.rag_pipeline import RAGClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.utils import tokenize

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,)

pdf = read_pdf('data\printsipy-proektirovaniya-integralnoy-modeli-otsenki-nadezhnosti-informatsionno-vychislitelnyh-sistem.pdf')
chunks = text_splitter.split_text(pdf)

rag = RAGClient(db_path='./db', collection_name='docs')
rag.add_documents(chunks)


tokenized_corpus = [tokenize(doc) for doc in chunks]
bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

with open('bm25_index.pkl', 'wb') as f: 
    data = {
    'model': bm25,
    'chunks': chunks
    }
    pickle.dump(data, f)