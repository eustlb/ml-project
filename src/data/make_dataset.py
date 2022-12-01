# libraries
import os
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from definitions import ROOT_DIR
from pathlib import Path


def import_clean_data(dataset_name):
    """
    Import data and replace NaN values by mean value of the corresponding feature.

    :param dataset_name: str, dataset_name, should be one of ["prostate.data.txt", "HousingData.csv"]
    :return: pandas DataFrame, new dataframe with NaN values replaced by mean value of the corresponding feature
    """
    data_path = os.path.join(ROOT_DIR, "data/raw", dataset_name)
    extension = Path(data_path).suffix
    if extension == ".csv":
        data = pd.read_csv(data_path)
    elif extension == ".txt":
        f = open(data_path, "r")
        file = f.read()
        f.close()
        columns = file.split("\n")[0].split()
        d = [line.split()[1:] for line in file.split("\n")[1:-1]]
        data = pd.DataFrame(data=d, columns=columns)
        # convert string cols
        for col in data.columns:
            if col != "train":
                data[col] = data[col].astype(float)

    # replace missing values by mean
    new_data = data.fillna(data.mean())
    return new_data


def standardizer(data):
    """
    Standardize each feature of the data.

    :param data: pandas DataFrame, input dataframe that is to be standardized with features as columns.
    :return: pandas DataFrame.
    """
    for col in data.columns:
        if col != 'train':
            data[col] = (data[col]-data[col].mean())/data[col].std()
    return data


def split_data_train_test(data, label_column_name, test_size=0.3, random_state=42):
    """
    Split data into train set and test set.

    :param data: pandas DataFrame, input dataframe that is to be splited with features as columns.
    :param label_column_name: str, name of the targeted label.
    :param test_size: float, test size as a percentage eg 0.3.
    :param random_state: int, to get reproducible results.
    :return: tuple, train-test split of inputs.
    """
    if "train" in data.columns:
        data_train = data[data.train == "T"].drop("train", axis=1)
        data_test = data[data.train == "F"].drop("train", axis=1)
        X_train, y_train = data_train.loc[:, data_train.columns != label_column_name], data_train[label_column_name]
        X_test, y_test = data_test.loc[:, data_test.columns != label_column_name], data_test[label_column_name]

    else:
        features = data.loc[:, data.columns != label_column_name]
        targets = data[label_column_name]
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size,
                                                            random_state=random_state)

    return X_train, X_test, y_train, y_test


