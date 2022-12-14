# libraries
import os
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import train_test_split
from definitions import ROOT_DIR
from pathlib import Path
from sklearn.decomposition import PCA


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
    new_data = data.fillna(data.mean(numeric_only=True))
    return new_data


def standardizer(data):
    """
    Standardize each feature of the data.

    :param data: pandas DataFrame, input dataframe that is to be standardized with features as columns.
    :return: pandas DataFrame.
    """
    for col in data.columns:
        if col != 'train':
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data


def feature_select(data, label_column_name, var_thresh=0.95):
    """
    This function does feature selection on the data using PCA. We set a threshold on the pourcentage of variance that
    should be explained by the selected components. That is how we determine which components to keep.
    :param data: Pandas dataframe containing input data
    :param label_column_name: Name of the column that contains labels
    :param var_thresh: Threshold to use on the percentage of variance explained by the selected components. It should be
     a number between 0 and 1.Default is 0.95 .
    :return: Pandas dataframe containing the selected features and labels.
    """

    if "train" in data.columns:
        labels = data[[label_column_name, 'train']]
        data_no_label = data.drop([label_column_name, 'train'], axis=1)
    else:
        labels = data[label_column_name]
        data_no_label = data.drop(label_column_name, axis=1)
    
    pca = PCA()
    pca.fit(data_no_label)
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    n_features = np.shape(data_no_label)[1] - sum(explained_var >= var_thresh)
    new_data = pca.fit_transform(data_no_label)[:, :n_features]
    return pd.concat([pd.DataFrame(new_data), labels], axis=1)


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
