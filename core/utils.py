import re

def tokenize(text):
    tokenized_text = text.lower()
    tokenized_text = re.findall(r'\w+', tokenized_text)
    return tokenized_text