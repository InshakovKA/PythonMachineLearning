import numpy as np

# Пример реализации эффективных векторизованных функций с помощью numpy


def sum_non_neg_diag(X: np.ndarray) -> int:  # Сумма неотрицательных элементов на диагонали прямоугольной матрицы
    d = np.diag(X)
    if not np.any(d >= 0):
        return -1
    non_neg = np.where(d >= 0, d, 0)
    return non_neg.sum()


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:  # Сравнение двух мультимножеств
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:  # Максимальное прозведение соседних элементов в массиве x,
    # таких что хотя бы один множитель в произведении делится на 3
    y = np.roll(x, 1)
    m = x * y
    m = np.delete(m, [0])
    m = np.delete(m, np.where(m % 3 != 0))
    return np.amax(m) if len(m) else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:  # Сложение каналов двухмерного изображения
    # с указанными весами.
    return np.dot(image, weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:  # скалярное произведение между векторами x и y,
    # заданными в формате RLE
    count_x = x[:, 1]
    value_x = x[:, 0]
    count_y = y[:, 1]
    value_y = y[:, 0]
    if np.sum(count_x) != np.sum(count_y):
        return -1
    full_x = np.repeat(value_x, count_x)
    full_y = np.repeat(value_y, count_y)
    return np.dot(full_x, full_y)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:  # Вычисление матрицы косинусных расстояний
    # между объектами X и Y
    prod = np.matmul(X, Y.transpose())
    mod_x = np.apply_along_axis(norm, 1, X)
    mod_y = np.apply_along_axis(norm, 1, Y)
    mod_y = np.atleast_2d(mod_y)
    tmp = np.atleast_2d(mod_x).transpose()
    mod = np.matmul(tmp, mod_y)
    res = np.divide(prod, mod, out=np.ones_like(prod, dtype=np.float64), where=mod != 0)
    return res


def dist(x: np.ndarray, y: np.ndarray):  # Косинусное расстояние между векторами x и y
    mod_x = np.sum(x * x) ** 0.5
    mod_y = np.sum(y * y) ** 0.5
    if (mod_x == 0) or (mod_y == 0):
        return 1.0
    return np.dot(x, y) / (mod_x * mod_y)


def norm(x):  # Норма вектора x
    return np.sum(x * x) ** 0.5
