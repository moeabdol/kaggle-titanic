from sklearn.ensemble import RandomForestRegressor

# utility function to fill in missing age values
def predict_missing_ages(df):
    temp_df = df[['Age', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp',
        'Master', 'Miss', 'Mr', 'Mrs', 'C', 'Q', 'S']]
    known_ages_df = temp_df[temp_df['Age'].notnull()]
    unknown_ages_df = temp_df[temp_df['Age'].isnull()]
    X = known_ages_df.values[:, 1:]
    y = known_ages_df.values[:, 0]

    # use random forest to predict missing ages
    clf = RandomForestRegressor(n_estimators=2000, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(unknown_ages_df.values[:, 1:])

    df.loc[df['Age'].isnull(), 'Age'] = y_pred
