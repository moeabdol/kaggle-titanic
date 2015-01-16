import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# function to predict survival of passengers in the test dataset
def predict_survival(train_df, test_df):
    # read best hyperparameters from csv
    params = pd.read_csv('./temp/best_params.csv', header=0)
    params = params.where(pd.notnull(params), other=None)   # replace nan
    n_estimators = params.n_estimators.values[0]
    criterion = params.criterion.values[0]
    max_features = params.max_features.values[0]
    max_leaf_nodes = params.max_leaf_nodes.values[0]
    min_samples_leaf = params.min_samples_leaf.values[0]
    min_samples_split = params.min_samples_split.values[0]

    # prepare data and fit classifier
    X = train_df.values[:, :-1]
    y = train_df.values[:, -1]
    clf = RandomForestClassifier(n_estimators=n_estimators,
        criterion=criterion, max_features=max_features,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, n_jobs=-1, random_state=42)
    clf.fit(X, y)

    # make prediction
    X_test = test_df.values
    y_pred = clf.predict(X_test).astype(int)

    # write prediction to csv
    with open('./result/result.csv', 'wt') as f:
        w = csv.writer(f)
        w.writerow(['PassengerId', 'Survived'])
        for i in range(892, 1310):
            w.writerow([i, y_pred[i - 892]])
