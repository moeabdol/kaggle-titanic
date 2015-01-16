from sklearn.preprocessing import StandardScaler

# utility function to scale (normalize) features in the dataset
def scale_features(df):
    features = ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp']
    temp_df = df[features]
    scalar = StandardScaler()
    df[features] = scalar.fit_transform(temp_df)
