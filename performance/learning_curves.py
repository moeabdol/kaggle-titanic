import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

# method to plot learning curves
def plot_learning_curves(df):
    # get learning curves
    X = df.values[:, :-1]
    y = df.values[:, -1]
    clf = RandomForestClassifier(n_estimators=n_estimators,
        criterion=criterion, max_features=max_features,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, n_jobs=-1, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=10,
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # get mean and std deviation
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot learning curves
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Train')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Test')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, color='b', alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, color='r', alpha=0.1)
    plt.title("Random Forest Classifier")
    plt.legend(loc='best')
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.ylim(0.6, 1.01)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.draw()
    plt.savefig('./figures/learning_curves.png')
    plt.clf()

# method to plot ROC curve
def plot_ROC_curve(df):
    X = df.values[:, :-1]
    y = df.values[:, -1]
    clf = RandomForestClassifier(n_estimators=n_estimators,
        criterion=criterion, max_features=max_features,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, n_jobs=-1, random_state=42)

    # get fpr and tpr
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
            random_state=42)
    clf.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])

    # calculate auc
    roc_auc = auc(fpr, tpr)

    # plot ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.draw()
    plt.savefig('./figures/roc_curve.png')
    plt.clf()

# read best hyperparameters from csv
params = pd.read_csv('./temp/best_params.csv', header=0)
params = params.where(pd.notnull(params), other=None)   # replace nan
n_estimators = params.n_estimators.values[0]
criterion = params.criterion.values[0]
max_features = params.max_features.values[0]
max_leaf_nodes = params.max_leaf_nodes.values[0]
min_samples_leaf = params.min_samples_leaf.values[0]
min_samples_split = params.min_samples_split.values[0]

