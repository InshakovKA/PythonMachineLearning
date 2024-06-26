import numpy as np

# Реализация двух видов нормализации данных.


class MinMaxScaler:
    def __init__(self):
        self.maxs = np.empty([1, 1])
        self.mins = np.empty([1, 1])

    def fit(self, data: np.ndarray) -> None:
        """
        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        """
        self.maxs = data.max(axis=0)
        self.mins = data.min(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        :param np.ndarray data: Непустой двумерный массив векторов-признаков
        :return np.ndarray: Исходные данные масштабированные на отрезок [0, 1]
        """
        for i in range(data.shape[1]):
            data[:, i] -= self.mins[i]
            data[:, i] /= self.maxs[i] - self.mins[i]
        return data


class StandardScaler:
    def __init__(self):
        self.means = np.empty([1, 1])
        self.divs = np.empty([1, 1])

    def fit(self, data: np.ndarray) -> None:
        """
        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        """
        self.means = data.mean(axis=0)
        self.divs = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        :param np.ndarray data: Непустой двумерный массив векторов-признаков
        :return np.ndarray: Исходные данные масштабированные на нормальное распределение
        """
        for i in range(data.shape[1]):
            data[:, i] -= self.means[i]
            data[:, i] /= self.divs[i]
        return data
