import pandas as pd
from features.sex import binarize_sex
from features.name import create_title_columns
from features.embarked import create_embarked_columns
from features.age import predict_missing_ages
from features.scale import scale_features
from features.interaction_features import create_interaction_features

# utility function to clean, rearrange, merge and preprocess  train and test
# data
def preprocess_data(train_df, test_df):
    # merge train_df and test_df into one dataframe for later manipulation
    train_shape = train_df.shape
    df = pd.concat([train_df, test_df], axis=0)

    # change sex values to binary values 0 female, 1 male
    binarize_sex(df)

    # create dummy title columns (Mr, Master, Mrs, Miss)
    df = create_title_columns(df)

    # create dummy embarked columns (C, Q, S)
    df = create_embarked_columns(df)

    # fill in the missing fare value
    df.loc[df['Fare'].isnull(), 'Fare'] = \
        df[df['Pclass'] == 3]['Fare'].mode()[0]

    # fill in missing age values
    predict_missing_ages(df)

    # scale features
    scale_features(df)

    # drop all unecessary columns
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Title', axis=1, inplace=True)
    df.drop('Embarked', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)

    df.reset_index(inplace=True)                # reset index of the merged df
    df.drop('index', axis=1, inplace=True)
    df.reindex_axis(df.columns, axis=1)         # reindex columns

    # create interaction features
    df = create_interaction_features(df)

    # rearrange the dataframe by moving survived column to the end
    cols = df.columns.tolist()
    cols = cols[:6] + cols[7:] + [cols[6]]
    df = df[cols]

    # split dataframe to train_df and test_df dataframes
    train_df = df.iloc[:train_shape[0], :]
    test_df = df.iloc[train_shape[0]:, :-1]

    return train_df, test_df
