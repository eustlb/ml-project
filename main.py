# imports
import os
from src.data.make_dataset import import_clean_data, normalizer, features_target, split_data_train_test
from src.models.train_model import train_model

# choose dataset
dataset_name = "HousingData"
label_column_name = "MEDV"

# load and preprocess data
data_path = os.path.join(os.path.dirname(__file__), "data/raw", dataset_name+".csv")
data_df = import_clean_data(data_path)
norm_data_df = normalizer(data_df)

# train/test split
test_size = 0.3
X_train, X_test, y_train, y_test = split_data_train_test(norm_data_df, dataset_name, label_column_name, test_size)
print(type(X_train))

# train model
model_name = "LinearRegressor"
trained_model = train_model(model_name, {}, X_train, y_train, 10)




