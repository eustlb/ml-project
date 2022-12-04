# imports
import os
from src.data.make_dataset import import_clean_data, standardizer, split_data_train_test, feature_select
from src.models.train_model import train_model
from sklearn.metrics import mean_squared_error

# choose dataset
dataset_name = "HousingData.csv"
#dataset_name = "prostate.data.txt"
label_column_name = "MEDV"

# load and preprocess data
data_df = import_clean_data(dataset_name)
std_data_df = standardizer(data_df)
selected_data_df = feature_select(std_data_df, label_column_name, 0.95)

# train/test split
test_size = 0.3
X_train, X_test, y_train, y_test = split_data_train_test(selected_data_df, label_column_name, test_size)

# train model
model_name = "LinearRegressor"
trained_model = train_model(model_name, {}, X_train, y_train, 10)

# validate model
predictions = trained_model.predict(X_test)
score = mean_squared_error(y_test, predictions)
print(score)
