import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Тестирование алгоритма KMeans на различных данных


def main():
    colors_clusters = ['g', 'b', 'r', 'y']
    k = [2, 4, 3, 3, 4, 3]
    for i in range(1, 7):
        data = pd.read_csv(f"{i}.csv", index_col=None)
        if i in [1, 2, 3, 5]:
            a_x = data.iloc[:, 0]
            a_y = data.iloc[:, 1]
        else:
            a_x = data.iloc[:, 1]
            a_y = data.iloc[:, 2]
        X = [[a_x[j], a_y[j]] for j in range(len(a_x))]
        model = KMeans(n_clusters=k[i-1], random_state=42, n_init='auto').fit(X)
        plt.figure(figsize=(10, 10))
        for m in range(len(X)):
            plt.scatter(X[m][0], X[m][1], c=colors_clusters[model.labels_[m]])
        plt.show()


if __name__ == "__main__":
    main()