import numpy as np
from sklearn.tree import DecisionTreeClassifier
np.random.seed(42)

# Реализация случайного леса на основе существующей модели решающего дерева


class sample(object):
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


class random_forest(object):
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        self.subspace_idx = []
        self._estimators = []
        self.subspace_idx = []

    def fit(self, X, y):
        split = sample(X, self.subspaces_dim)
        for i in range(self.n_estimators):
            subsp = split.random_subspace(X, self.subspaces_dim)
            x_ind = split.bootstrap_sample(X)
            self.subspace_idx.append(subsp)
            new_x, new_y = split.get_subsample(X, y, subsp, x_ind)
            tmp = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tmp.fit(new_x, new_y)
            self._estimators.append(tmp)

    def predict(self, X):
        res = []
        for x in X:
            votes = []
            for i in range(self.n_estimators):
                tree = self._estimators[i]
                votes.append(tree.predict(x[self.subspace_idx[i]].reshape(1, -1)))
            votes = np.array(votes).reshape(1, -1)[0]
            vals, counts = np.unique(votes, return_counts=True)
            mode_value = np.argwhere(counts == np.max(counts))
            vote = votes[mode_value[0]]
            res.append(vote[0])
        return np.array(res)