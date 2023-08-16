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


def cut_matrix(field: np.array, params: np.array) -> np.array:
    """Функция выборки матрицы по указанным столбцам и строкам

    Args:
        field (np.array): исходная матрица (временная карта)
        params (np.array): массив с указанием вырезаемых столбцов и строк

    Returns:
        np.array: результирующая матрица (временная карта)
    """
    length = field.shape[0]
    row_indexes = np.where(params[:length]==1)[0]
    col_indexes = np.where(params[length:]==1)[0]
    if row_indexes.size==0:
        return field[:, col_indexes]
    elif col_indexes.size==0:
        return field[row_indexes, :]
    else:
        tmp = cut_matrix[row_indexes, :]
        return tmp[:, col_indexes]

