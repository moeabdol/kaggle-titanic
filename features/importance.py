import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# utility function to plot features importance
def plot_feature_importances(important_features, important_values):
    yticks = np.arange(len(important_features)) + 0.5
    plt.figure()
    plt.barh(yticks, important_values[::-1], align='center')
    plt.yticks(yticks, important_features[::-1])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.draw()
    plt.savefig('./figures/featrue_importances.png')
    plt.clf()

# function to find and use only most important features
def use_most_important_features(train_df, test_df, importance_threshold=15):
    # random forest classifier can identify most important features
    # train the classifier and get features importance
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    clf = RandomForestClassifier(n_estimators=10000, oob_score=True,
            random_state=42)
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    feature_importances = 100.0 * (feature_importances /
            feature_importances.max())

    # arrange most important features in descending order
    feature_list = train_df.columns.values[:-1]
    important_idx = np.where(feature_importances > importance_threshold)[0]
    important_features_names = feature_list[important_idx]
    important_features_values = feature_importances[important_idx]
    sorted_idx = np.argsort(important_features_values)[::-1]
    important_features_names = important_features_names[sorted_idx]
    important_features_values = important_features_values[sorted_idx]

    # plot most important features
    #plot_feature_importances(important_features_names,
            #important_features_values)

    # rearrange train_df and test_df to include only most important features
    train_df = train_df[np.concatenate((important_features_names,
        ['Survived']))]
    test_df = test_df[important_features_names]

    return train_df, test_df
