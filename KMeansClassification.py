import numpy as np
import sklearn
from sklearn.cluster import KMeans

# Реализация классификации частично размеченных данных на основе алгоритма кластериазции KMeans и метода самообучения.


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        """
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster = KMeans(n_clusters=self.n_clusters)
        self.translate = []

    def fit(self, data, labels):
        """
        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки.
        Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
        :return KMeansClassifier
        """
        self.cluster.fit(data)
        mapping, predicted = self._best_fit_classification(self.cluster.labels_, labels)
        self.translate = mapping
        return self

    def predict(self, data):
        """
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        """
        clusters = self.cluster.predict(data)
        predictions = np.take(self.translate, clusters)
        return predictions

    def _best_fit_classification(self, cluster_labels, true_labels):
        """
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов
        """
        setlables, counts = np.unique(np.delete(true_labels, np.argwhere(true_labels == -1)), return_counts=True)
        mapping = []
        for cluster in range(self.n_clusters):
            cluster_labeled = np.delete(true_labels, np.argwhere(cluster_labels != cluster))
            cluster_labeled = np.delete(cluster_labeled, np.argwhere(cluster_labeled == -1))
            if not len(cluster_labeled):
                biggest = setlables[np.argmax(counts)]
                mapping.append(biggest)
            else:
                cluster_counts = np.bincount(cluster_labeled)
                mapping.append(np.argmax(cluster_counts))
        predicted_labels = np.take(mapping, cluster_labels)
        return mapping, predicted_labels
