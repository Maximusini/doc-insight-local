import pypdf

def read_pdf(file_path):
    reader = pypdf.PdfReader(file_path)
    text = []
    
    for page in reader.pages:
        text.append(page.extract_text())
        
    full_text = '\n'.join(text)
    
    return full_text