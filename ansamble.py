import numpy as np

# Получение подвыборки для бутстрэппинга


class Sample(object):
    def __init__(self, X, n_subspace):
        self.idx_subspace = self.random_subspace(X, n_subspace)

    def __call__(self, X, y):
        idx_obj = self.bootstrap_sample(X)
        X_sampled, y_sampled = self.get_subsample(X, y, self.idx_subspace, idx_obj)
        return X_sampled, y_sampled

    @staticmethod
    def bootstrap_sample(X, random_state=42):
        res = np.random.choice(X.shape[0], size=X.shape[0])
        return np.unique(res)

    @staticmethod
    def random_subspace(X, n_subspace, random_state=42):
        res = np.random.choice(X.shape[1], size=n_subspace, replace=False)
        return res

    @staticmethod
    def get_subsample(X, y, idx_subspace, idx_obj):
        return X[np.ix_(idx_obj, idx_subspace)], y[idx_obj]
