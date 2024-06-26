import numpy as np

# Реализация алгоритма KMeans без использования готовых решений


def dist(x1, x2):
    s = 0
    for i in range(len(x1)):
        s += (x1[i] - x2[i]) ** 2
    return s ** 0.5


class KMeans(object):
    def __init__(self, K, init):
        self.K = K
        self.centers = init
        self.dim = len(self.centers[0])

    def fit(self, X):
        prev = self.centers
        flag = True
        while flag:
            classes = dict.fromkeys(range(self.K), [])
            for x in X:
                distances = np.array([dist(x, c) for c in self.centers])
                cla = np.argmin(distances)
                classes[cla].append(x)
            self.centers = []
            for i in range(self.K):
                points = [0] * self.dim
                size = 0
                for x in classes[i]:
                    size += 1
                    for j in range(len(x)):
                        points[j] += x[j]
                center = []
                for p in points:
                    center.append(p / size)
                self.centers.append(center)
            print(self.centers)
            diff = []
            for i in range(self.K):
                diff.append(dist(self.centers[i], prev[i]))
            if max(diff) <= 0.001:
                flag = False
            prev = self.centers

    def predict(self, X):
        res = []
        for x in X:
            ans = 0
            min_dist = dist(self.centers[0], x)
            for i in range(len(self.centers)):
                tmp = dist(self.centers[i], x)
                if tmp < min_dist:
                    min_dist = tmp
                    ans = i
            res.append(ans)
        return np.array(res)
