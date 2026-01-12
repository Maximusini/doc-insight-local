FROM python:3.12-slim

WORKDIR ./app

RUN pip install poetry

COPY ./poetry.lock ./pyproject.toml ./

RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]