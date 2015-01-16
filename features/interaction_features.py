import pandas as pd

# utility function to create interaction features with (*, /, +, -) operations
def create_interaction_features(df):
    cols = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']
    cols_len = len(cols)
    temp_df = df[cols]
    for i in range(cols_len):
        coli = temp_df.columns[i]
        for j in range(cols_len):
            colj = temp_df.columns[j]
            if i < j:
                name = coli + '*' + colj
                df = pd.concat([df, pd.Series(temp_df.iloc[:, i] *
                    temp_df.iloc[:, j], name=name)], axis=1)
                name = coli + '/' + colj
                df = pd.concat([df, pd.Series(temp_df.iloc[:, i] /
                    temp_df.iloc[:, j], name=name)], axis=1)
                name = coli + '+' + colj
                df = pd.concat([df, pd.Series(temp_df.iloc[:, i] +
                    temp_df.iloc[:, j], name=name)], axis=1)
                name = coli + '-' + colj
                df = pd.concat([df, pd.Series(temp_df.iloc[:, i] -
                    temp_df.iloc[:, j], name=name)], axis=1)
    return df
