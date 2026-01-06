import re
import os
import shutil
from core.config import *

def tokenize(text):
    tokenized_text = text.lower()
    tokenized_text = re.findall(r'\w+', tokenized_text)
    
    return tokenized_text


def save_uploaded_file(uploaded_file):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path


def clear_database():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename != '.gitkeep':
                file = os.path.join(DATA_DIR, filename)
                os.remove(file)