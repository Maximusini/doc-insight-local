import ollama
from core.rag_pipeline import RAGClient

rag = RAGClient()

def search_kb(query:str):
    print(f'Ищу в базе: {query}')
    return rag.query(query)

def add_two_numbers(a: int, b: int) -> int:
    return int(a) + int(b)


tools = [
    {
        'type': 'function',
        'function': {
            'name': 'add_two_numbers',
            'description': 'Add two integers together',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {
                        'type': 'integer',
                        'description': 'The first number',
                    },
                    'b': {
                        'type': 'integer',
                        'description': 'The second number',
                    },
                },
                'required': ['a', 'b'],
            },
        },
    },
    
    {
        'type': 'function',
        'function': {
            'name': 'search_knowledge_base',
            'description': 'Use this tool to find information in documents',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The question',
                    }
                },
                'required': ['query']
            },
        },
    },
]

available_functions = {
        'add_two_numbers': add_two_numbers,
        'search_knowledge_base': search_kb
    }

response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Какие модели надежности существуют?'}], tools = tools)
if response['message'].get('tool_calls'):
    for tool in response['message']['tool_calls']:
        
        func_name = tool['function']['name']
        if func_name in available_functions:
            args = tool['function']['arguments']
            result = available_functions[func_name](**args)
            print(f'Результат выполнения: {result}')
            
            final_answer = ollama.chat(model='llama3.2',
                                       messages=[
                                           {'role': 'system', 'content': 'Ты эксперт. Используй найденные данные, чтобы ответить на вопрос пользователя на русском языке.'},
                                           {'role': 'user', 'content': f'Вопрос был: Какие модели надежности существуют? \n\n Найденные данные: {result}'}
                                       ])
            print('\nИтоговый ответ Агента:\n')
            print(final_answer['message']['content'])