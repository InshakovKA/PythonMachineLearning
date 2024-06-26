import numpy as np
from sklearn.metrics import pairwise_distances

# Реализация коэффициента силуэта.


def silhouette_score(x, labels):
    """
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    """
    sil_score = 0
    k = 0
    n = len(labels)
    setlabels, counts = np.unique(labels, return_counts=True)
    dist = pairwise_distances(x)
    labelind = dict.fromkeys(setlabels)
    means = []
    for j in setlabels:
        labelind[j] = k
        k += 1
        means.append(np.mean(dist, axis=1, where=(labels == j)))
    inds = np.array(list(labelind.values()))
    classmean = np.column_stack(means)
    if classmean.shape[1] == 1:
        return 0
    for i in range(n):
        label = labels[i]
        sil = 0
        ind = labelind[label]
        if counts[ind] > 1:
            a = classmean[i][ind] * counts[ind] / (counts[ind] - 1)
            b = np.min(classmean[i], where=(inds != ind), initial=2**31)
            if a != 0 and b != 0:
                sil = (b - a) / max(a, b)
        sil_score += sil
    return sil_score / n
