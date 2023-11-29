import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from Assets import load_data, load_test_Data
from sklearn.ensemble import RandomForestClassifier as RTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV as grid
from sklearn.model_selection import PredefinedSplit


class myRanForest:
    def __init__(self):
        pass

    def find_best(self):
        train, validation, test = load_data()
        X_train, y_train = train
        X_val, y_val = validation
        X_test, y_test = test
        split_index = [
            -1 if x in X_train.index else 0 for x in pd.concat([X_train, X_val]).index
        ]
        pds = PredefinedSplit(split_index)
        param_grid = {
            "n_estimators": [100, 500, 1000, 2000],
            "criterion": ["entropy", "gini"],
            "max_depth": [2, 4, 6, 10, 14, 20],
        }
        clf = grid(
            estimator=RTree(random_state=42), cv=pds, param_grid=param_grid, verbose=3
        )
        # Fit with all data
        clf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        model = RTree(**clf.best_params_, random_state=42).fit(X_train, y_train)
        train_score = accuracy_score(model.predict(X_train), y_train) * 100
        val_score = accuracy_score(model.predict(X_val), y_val) * 100
        test_score = accuracy_score(model.predict(X_test), y_test) * 100
        print("Score for train set:", train_score)  # 73.86%
        print("Score for validation set:", val_score)  # 61.8%
        print("Score for test set:", test_score)  # 64.96
        print(
            "This is with the 'best' params of:", clf.best_params_
        )  # entropy, estimators=2000 max_depth=6
        return model

    def produce_best(self):
        train, validation, test = load_data()
        X_train, y_train = train
        X_val, y_val = validation
        X_test, y_test = test
        model = RTree(
            criterion="entropy", n_estimators=2000, max_depth=6, random_state=42
        ).fit(X_train, y_train)
        train_score = accuracy_score(model.predict(X_train), y_train) * 100
        val_score = accuracy_score(model.predict(X_val), y_val) * 100
        test_score = accuracy_score(model.predict(X_test), y_test) * 100
        print("Score for train set:", train_score)  # 73.86%
        print("Score for validation set:", val_score)  # 61.8%
        print("Score for test set:", test_score)  # 64.96
        return model


if __name__ == "__main__":
    mrf = myRanForest()
    model = mrf.produce_best()
    data = load_test_Data()
    preds = list(model.predict(data))
    print([str(num) + ":" + str(preds.count(num)) for num in set(preds)])
    pd.DataFrame(preds).to_csv("output", index=False, header=False)
