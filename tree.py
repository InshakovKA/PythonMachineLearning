import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
np.random.seed(42)

# Сравнение работы решающих деревьев с различными параметрами


def main():
    data = pd.read_csv("TRAIN.csv")
    data = data.drop(columns="Unnamed: 0")
    le = LabelEncoder()
    data["cut"] = le.fit_transform(data["cut"])
    data["color"] = le.fit_transform(data["color"])
    data["clarity"] = le.fit_transform(data["clarity"])
    data = shuffle(data)
    no_price = data.drop(columns="price")
    print(data)
    depths = [12, 16, 22, 45, 95, 33]
    criteria = ["squared_error", "friedman_mse", "poisson", "squared_error", "friedman_mse", "poisson"]
    for i in range(6):
        tree = DecisionTreeRegressor(max_depth=depths[i], criterion=criteria[i])
        print(f"{i}: {np.mean(cross_val_score(tree, no_price, data['price'], scoring='r2', cv=10))}")


if __name__ == "__main__":
    main()