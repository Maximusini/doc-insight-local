import json
import ollama
from core.rag_pipeline import RAGClient

with open('eval_dataset.json', 'rb') as f:
    eval_dataset = json.load(f)
rag = RAGClient()


correct_answers = 0

for data in eval_dataset:
    context = rag.query(data['question'])
    response = rag.generate_answer(context, data['question'])
    print(response)
    
    prompt = f'''Сравни ответ кандидата с эталоном. Если смысл совпадает - верни 1. Если нет - верни 0.
                 Ответ кандидата: {response}
                 Эталон: {data['ground_truth']}'''
    
    judge = ollama.chat(model='gemma3:4b', messages=[{'role': 'user', 'content': prompt}])
    print(judge['message']['content'])
    
    if '1' in judge['message']['content']:
        correct_answers += 1
        
accuracy = correct_answers / len(eval_dataset)
print(accuracy)