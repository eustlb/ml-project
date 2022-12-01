from src.data.make_dataset import import_clean_data, standardizer

datasets_names = ["prostate.data.txt", "HousingData.csv"]

# verify cleaning and standardization of the data
for datasets_name in datasets_names:
    # clean the dataset
    df = import_clean_data(datasets_name)
    print(f"Number of NaN values (should be 0) : {df.isna().sum().sum()}")
    # standardize each feature
    df = standardizer(df)
    print(df.describe().loc[['mean', 'std']])




