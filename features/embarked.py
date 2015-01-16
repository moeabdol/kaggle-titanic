import pandas as pd

# utility function to create dummy embarked columns (C, Q, S)
def create_embarked_columns(df):
    df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
    return df
