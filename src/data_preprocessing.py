# src/data_preprocessing.py
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Fetch the California Housing dataset
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

# This data does not contain missing values, but if there were
def clean_data(df):
    df = df.dropna()
    return df

# Handling categorical values requires use of OneHotEncoder or OrdinalEncoder


