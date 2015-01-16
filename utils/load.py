import pandas as pd
from os.path import join

# utility function to load data files into two pandas dataframes
def load_data(directory):
    train_df = pd.read_csv(join(directory, 'train.csv'), header=0)
    test_df = pd.read_csv(join(directory, 'test.csv'), header=0)
    return train_df, test_df
