import numpy as np


def get_min(field: np.array, row: bool = True) -> np.array:
    """Получение минимальных значение временной карты

    Args:
        field (np.array): временная карта
        row (bool, optional): поиск по строкам (True) или по столбцам (False). Defaults to True.

    Returns:
        np.array: массив минимальных значений
    """
    return field.min(axis=int(row))


def rounding(field: np.array, rounding_value: int) -> np.array:
    """Округление значений карты до фиксированного интервала

    Args:
        field (np.array): временная карта
        rounding_value (int): шаг округления

    Returns:
        np.array: результирующая карта
    """
    max_time = field.max()
    field[field <= rounding_value] = rounding_value
    for i in range(rounding_value, max_time, rounding_value):
        field[(field > i) & (field <= (i+rounding_value))] = i+rounding_value
    return field
