"""Основной модуль приложения генерации таблиц."""

import json

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.src.utils.storage import Storage
from backend.src.utils.graph_storage import GraphStorage
from backend.src.bayes_table_generation import BayesTableGenerater


EXTENSION = '.json'

storage = Storage()
graph_storage = GraphStorage()

app = FastAPI()


app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get('/')
async def get_index():
    """Получение главной страницы."""
    return FileResponse('frontend/index.html')


@app.post('/upload-data/')
async def upload_data(file: UploadFile = File(...)):
    """
    Загрузка данных.

    Загружаются данные о лекарствах, побочных действиях,
    и весах (рангах) их появления.
    """
    if not file.filename.endswith(EXTENSION):
        return {'error': 'Файл с данными должен быть формата JSON'}

    content = await file.read()

    try:
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError:
        return {'error': 'Некорректный JSON'}

    storage.save(data)

    return {'message': ('Данные о лекарствах, побочных дествиях'
                        ' и их весах загружены успешно!')}


@app.post('/upload-graph-and-propabilities/')
async def upload_graph_and_propabilities(
        graph_file: UploadFile = File(...),
        propability_file: UploadFile = File(...)):
    """Загрузка графа и вероятностей."""
    if (not graph_file.filename.endswith(EXTENSION)
            or not propability_file.filename.endswith(EXTENSION)):
        return {'error': 'Оба файла должны быть в формате JSON'}

    graph_content = await graph_file.read()
    propability_content = await propability_file.read()

    try:
        graph = json.loads(graph_content.decode('utf-8'))
        propabilities = json.loads(propability_content.decode('utf-8'))
    except json.JSONDecodeError:
        return {'error': 'Некорректные JSON'}

    graph_storage.save_graph(graph)
    graph_storage.save_probability(propabilities)

    return {'message': 'Семантический граф и вероятности загружены успешно!'}


@app.get('/generate-bayes-table/')
async def generate_bayes_table():
    """Генерация таблицы для сети Байеса."""
    return Response(
        content=BayesTableGenerater().generate_bayes_table(),
        media_type=('application/vnd.openxmlformats-officedocument'
                    '.spreadsheetml.sheet'),
        headers={
            "Content-Disposition": "attachment; filename=bayes_report.xlsx"
        }
    )
