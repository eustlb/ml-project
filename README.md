# ml-project
Machine learning development project at IMT Atlantique.

# Project structure

    ml-project
    ├── data             
    │   └── raw                   <- Raw datasets are stored here.
    │
    ├── src 
    │   ├── __init__.py           <- Makes src a python module.    
    │   │
    │   ├── data          
    │   │   └── make_dataset.py   <- Script to preprocess datasets.
    │   │
    │   ├── models        
    │   │   └── train_model.py    <- Script to train a model.
    │   │
    │   └── visualization         
    │       └── performances.py    <- Script to train evaluate linear regression performances.
    │
    ├── tests                   
    │   └── make_dataset_test.py  <- Unit test to check data preprocessing.
    │
    ├── definitions.py            <- Define project variables such as ROOT_DIR.
    │
    ├── main.ipynb                <- Notebook to run the code.
    │
    ├── main.py                   
    │
    └── README.md

# Datasets
## Boston Housing
Downloaded from [Kaggle](https://www.kaggle.com/datasets/altavish/boston-housing-dataset) 

This dataset is composed of 506 observations of 14 variables.
There are 14 attributes in each case of the dataset:
- CRIM - per capita crime rate by town
- ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS - proportion of non-retail business acres per town.
- CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX - nitric oxides concentration (parts per 10 million)
- RM - average number of rooms per dwelling
- AGE - proportion of owner-occupied units built prior to 1940
- DIS - weighted distances to five Boston employment centres
- RAD - index of accessibility to radial highways
- TAX - full-value property-tax rate per $10,000
- PTRATIO - pupil-teacher ratio by town
- B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT - % lower status of the population
- **MEDV** - Median value of owner-occupied homes in $1000's

Here, we are trying to predict the variable **MEDV** from the 13 others. 

## Prostate cancer

This dataset is composed of 97 observations of 9 variables and an extra one being a train/test indicator.

There are 9 attributes:
- lcavol
- lweight 
- age
- lbph
- svi
- lcp
- gleason
- pgg45
- lpsa

Here, we are trying to predict the variable **lpsa** from the 13 others. 

# Ressources
[Machine learning project file structure.](https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices)
[Scikit-learn documentation.](https://scikit-learn.org/stable/)
