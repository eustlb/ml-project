# libraries
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split


# import data and file na with mean
def import_clean_data(data_path):
    """
    :param data_path:
    """
    data = pd.read_csv(data_path)
    # count na values
    # replace missing values by mean
    new_data = data.fillna(data.mean())
    return new_data


# normalization and centralization
def normalizer(data):
    data_normalized = data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return data_normalized


# Separate features and target variables
def features_target(data, label_column_name):
    features_columns = [column_name for column_name in data.columns if column_name != label_column_name]
    features, label = data[features_columns], data[[label_column_name]]
    return features, label


# Split data into train set and test set
def split_data_train_test(data, dataset:str, label_column_name, test_size=0.3, random_state=42):
    """
    :param data:
    :param dataset: st
    :param label_column_name:
    :param test_size:
    :param random_state:
    :return:
    """
    if dataset == 'prostate_cancer':
        label_column_name = data[data.columns[-1]]
        features_data, label = features_target(data, label_column_name)
        X_train, X_test, y_train, y_test = train_test_split(features_data, label,
                                                            random_state=random_state)
    else:
        features_data, label = features_target(data, label_column_name)
        X_train, X_test, y_train, y_test = train_test_split(features_data, label, test_size=test_size,
                                                            random_state=random_state)

    return X_train, X_test, y_train, y_test


