import streamlit as st
from core.rag_pipeline import RAGClient

st.set_page_config(page_title='Local RAG', page_icon='🤖')

@st.cache_resource
def get_rag():
    return RAGClient()

rag = get_rag()

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
        context = rag.query(prompt)
        response = rag.generate_answer(context, prompt)
    print(context)
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    with st.chat_message('assistant'):
        st.write(response)