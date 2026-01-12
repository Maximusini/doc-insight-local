import streamlit as st
import ollama
from core.rag_pipeline import RAGClient
from core.utils import *
from core.reader import read_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import *

st.set_page_config(page_title='Local RAG', page_icon='🤖')

@st.cache_resource
def get_rag(llm_model):
    return RAGClient(llm_model=llm_model)

try:
    models_list = [m['model'] for m in ollama.list()['models']]
except Exception:
    models_list = [LLM_MODEL]

with st.sidebar:
    model_selector = st.selectbox('Выбери модель', models_list)
    rag = get_rag(model_selector)
    
    file = st.file_uploader('Загрузи документ (PDF)', type='pdf')
    if file:
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != file.name:
            with st.spinner('Читаю и индексирую документ...'):
                file_path = save_uploaded_file(file)
                pdf = read_pdf(file_path)
                
                if len(pdf) < 50:
                    st.error('Файл пустой или это скан (картинка). Распознавание текста не поддерживается.')
                    if os.path.exists(file_path): os.remove(file_path)
                    st.stop()
                    
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                chunks = text_splitter.split_text(pdf)
                rag.build_indices(chunks)
                st.success('База знаний обновлена!')
                st.session_state.last_uploaded = file.name
                
    
    
    reset_btn = st.button('Очистить базу')
    if reset_btn:
        with st.spinner("Очищаю данные..."):
            rag.reset_database()
            
            clear_database()

            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
            
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

prompt = st.chat_input('Твой текст')

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)

    with st.spinner('Думаю...'):
        new_query = rag.contextualize_query(prompt, st.session_state.messages[:-1])
        st.write(f'🔄 *Ищу: {new_query}*')
        
        context_list = rag.query(new_query)
        
        if context_list:
            context = '\n---\n'.join(context_list)
            response = rag.generate_answer(context, new_query)
        else:
            response = 'Я пока ничего не знаю. Загрузи документ в меню слева!'
            context = 'Нет контекста'
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    with st.chat_message('assistant'):
        st.write(response)
        
    with st.expander(f'Источники (Найдено фрагментов: {len(context_list)})'):
        for i, doc in enumerate(context_list): 
            st.markdown(f'**Фрагмент #{i+1}**')
            st.info(doc)