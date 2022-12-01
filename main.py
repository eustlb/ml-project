# imports
import os
from src.data.make_dataset import import_clean_data, standardizer, split_data_train_test
from src.models.train_model import train_model

# choose dataset
# dataset_name = "HousingData.csv"
dataset_name = "prostate.data.txt"
label_column_name = "lpsa"

# load and preprocess data
data_df = import_clean_data(dataset_name)
std_data_df = standardizer(data_df)

# train/test split
test_size = 0.3
X_train, X_test, y_train, y_test = split_data_train_test(std_data_df, label_column_name, test_size)
print(X_train.head())
print(y_train.head())

# train model
# model_name = "LinearRegressor"
# trained_model = train_model(model_name, {}, X_train, y_train, 10)

#



