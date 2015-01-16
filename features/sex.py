# utility function to transform age column values to binary values 0 female,
# 1 male
def binarize_sex(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(float)
