from sklearn.preprocessing import LabelEncoder

def encode_labels(df):
    label = LabelEncoder()
    df['NEW_CUST'] = label.fit_transform(df['NEW_CUST'])
    df['SEX'] = label.fit_transform(df['SEX'])
    return df

def drop_unnecessary_columns(df):
    return df.drop('APP_ID', axis=1)

def define_features_and_target(df, target_column='STATUS'):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y
