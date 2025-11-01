import io

import pandas as pd
import numpy as np

from backend.src.utils.bayes_calculation import (load_combined_data,
                                                 get_result,
                                                 build_network,
                                                 calculate_probabilities)
from backend.src.utils.graph_storage import GraphStorage
from backend.src.utils.storage import Storage


class BayesTableGenerater:
    """Генератор таблиц для сети Байеса."""

    def __init__(self):
        """Создание генератора таблицы Байеса."""
        graph_storage = GraphStorage()
        self.name = 'name'
        self.side_effect_index = 0
        self.rank_index = 1
        self.data = Storage().download()
        self.graph = graph_storage.download_graph()
        self.probability_path = graph_storage.probability_path

    def _get_bin_ids(self, drugs):
        """Получение бинаризированного словаря."""
        bin_id = {}
        drug2id = {}

        for node in self.graph['nodes']:
            if node['label'] != 'prepare':
                continue
            drug2id[node[self.name]] = node['id']
            bin_id[node['id']] = 0

        for drug in drugs:
            bin_id[drug2id[drug.lower()]] = 1
        return bin_id

    def _get_side_effects(self):
        """Получение списка побочных действий."""
        effects = set()
        for drug in self.data:
            for pair in self.data[drug]:
                effects.add(pair[self.side_effect_index])
        return effects

    def _get_rank(self, drug, side_effect):
        """
        Получение ранга.

        Ранг ищется в хранилище данных
        для соответствующих лекарства и побочного действия.
        """
        for pair in self.data[drug]:
            if side_effect == pair[self.side_effect_index]:
                return pair[self.rank_index]

    def generate_bayes_table(self):
        """Получения таблицы рангов Байеса."""
        drugs = sorted(self.graph['name'])
        effects = sorted(self._get_side_effects())
        print('effects =', effects)
        bayes_table = pd.DataFrame('-', index=drugs, columns=effects)

        for drug in drugs:
            prob_data, drug_states_input_data, drugs_for_output, \
                combination_description = load_combined_data(
                    graph_data=self.graph,
                    prob_file=self.probability_path,
                    drug_states_input=self._get_bin_ids([drug])
                )

            # Построение сети с новыми параметрами
            network = build_network(self.graph, prob_data)

            # Перерасчет вероятностей (логирование не меняется)
            final_probs = calculate_probabilities(network)

            data = get_result(
                final_probs,
                self.graph,
                drug_states_input_data,
                drugs_for_output,
                combination_description)

            for effect in data['side_effects']:
                if effect in effects:
                    bayes_table.loc[drug, effect] = (
                        data['side_effects'][effect]['probability'])

        fortran_table = pd.DataFrame('-', index=drugs, columns=effects)
        for drug in drugs:
            for effect in effects:
                fortran_table.loc[drug, effect] = self._get_rank(drug, effect)

        # --- Преобразуем таблицы в числовой формат ---
        # Заменяем '-' на NaN и конвертируем в float
        bayes_numeric = bayes_table.replace('-', np.nan).astype(float)
        fortran_numeric = fortran_table.replace('-', np.nan).astype(float)

        # --- Вычисляем квадрат разницы ---
        diff_squared = (bayes_numeric - fortran_numeric) ** 2

        # Сумма по строкам (только числа)
        row_sums = diff_squared.sum(axis=1)

        # Заменяем NaN обратно на '-' для отображения (опционально)
        diff_squared_display = diff_squared.fillna('-')
        diff_squared_display['Сумма'] = row_sums.fillna('-')

        # Порог
        epsilon = 0.05

        # Абсолютная разница
        abs_diff = (bayes_numeric - fortran_numeric).abs()

        # Булево сравнение: True если разница <= epsilon
        accuracy_mask = abs_diff <= epsilon

        # Преобразуем True/False в "Верно"/"Неверно"
        accuracy_table = accuracy_mask.replace({True: 'Верно',
                                                False: 'Неверно'})

        # Восстанавливаем '-' вместо значений,
        # где хотя бы одна ячейка была пустой
        # (т.е. где была NaN в исходных числовых таблицах)
        mask_nan = bayes_numeric.isna() | fortran_numeric.isna()
        # Создаём копию accuracy_table
        accuracy_table = accuracy_table.copy()

        # Применяем маску напрямую: где mask_nan == True → ставим '-'
        accuracy_table[mask_nan] = '-'
        # Определяем, какие столбцы относятся к эффектам (все, кроме, возможно, уже добавленного 'Сумма')
        # В accuracy_table изначально столько же столбцов, сколько в effects
        effect_columns = effects  # или: [col for col in accuracy_table.columns if col != 'Сумма']

        # Извлекаем под-DataFrame только с эффектами
        acc_effect_part = accuracy_table[effect_columns]

        # Считаем количество "Верно" по строкам
        count_true = (acc_effect_part == 'Верно').sum(axis=1)

        # Считаем общее количество не-"-" значений (т.е. "Верно" или "Неверно")
        count_valid = acc_effect_part.isin(['Верно', 'Неверно']).sum(axis=1)

        # Вычисляем долю
        accuracy_ratio = count_true / count_valid

        # Где нет валидных данных — ставим NaN, затем заменим на '-'
        accuracy_ratio = accuracy_ratio.where(count_valid > 0, np.nan)

        # Добавляем столбец в основную таблицу
        accuracy_table['Доля верно'] = accuracy_ratio.fillna('-')

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            bayes_table.to_excel(writer, sheet_name='Байес', index=True)
            fortran_table.to_excel(writer, sheet_name='Фортран', index=True)
            diff_squared_display.to_excel(writer,
                                          sheet_name='Кв. разности',
                                          index=True)
            accuracy_table.to_excel(writer, sheet_name='Точность', index=True)

        excel_buffer.seek(0)

        return excel_buffer.getvalue()
