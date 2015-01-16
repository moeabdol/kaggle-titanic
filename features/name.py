import pandas as pd
import string

# list of titles to find in names
titles = ['Mr', 'Master', 'Miss', 'Mrs', 'Major', 'Ms', 'Mlle', 'Mme', 'Rev',
        'Dr', 'Col', 'Capt', 'Countess', 'Don', 'Jonkheer']

# utility function to find title in name
def title_in_name(name, titles):
    for title in titles:
        if string.find(name, title) != -1:
            return title
    return pd.NaN

# utility function to generalize titles
def generalize_titles(frame):
    title = frame['Title']
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme', 'Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms', 'Miss']:
        return 'Miss'
    elif title == 'Dr':
        if frame['Sex'] == 0:
            return 'Mrs'
        else:
            return 'Mr'
    else:
        return title

# utility function to create new column title and fill titles according to
# titles found in names
def create_title_columns(df):
    df['Title'] = df['Name'].map(lambda x: title_in_name(x, titles))
    df['Title'] = df.apply(generalize_titles, axis=1)
    df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1)

    return df
