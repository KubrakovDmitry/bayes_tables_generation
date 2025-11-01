"""Модуль хранения графа."""

import json
from pathlib import Path


class Storage:
    """
    Класс для хранения данных о ЛС, ПД и рангах.

    Данные храняться в директории data в JSON-файле.
    """

    DATA_FILE = 'data.json'
    DIR_PATH = Path('backend/src/data')

    def __init__(self):
        """Создание хранилища данных."""
        self.DIR_PATH.mkdir(exist_ok=True)
        self.data_path = self.DIR_PATH / self.DATA_FILE

        if not self.data_path.exists():
            f = open(self.data_path, 'w', encoding='utf-8')
            f.close()

    def save(self, data: dict):
        """Сохранение данных."""
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def download(self):
        """Выгрузка данных."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def size_of_graph_file(self):
        """Получение размера файла с данными."""
        return self.data_path.stat().st_size
