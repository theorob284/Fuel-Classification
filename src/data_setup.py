import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


NUMERIC_FEATURES = [
    "year",
    "engine_cylinders",
    "engine_displacement",
]

CATEGORICAL_FEATURES = [
    "class",
    "drive",
    "transmission",
    "turbocharger",
    "fuel_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "combined_l_per_100km"


def load_data(path="../processed_data/cleaned_fuel.csv"):
    df = pd.read_csv(path)
    cols_to_check = ['make', 'model'] + FEATURES + [TARGET]
    df = df.drop_duplicates(subset=cols_to_check).reset_index(drop=True)
    return df


def get_train_test_split(df, test_size=0.2, random_state=42):
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(X,y,test_size=test_size,random_state=random_state,)


def get_category_function(y_train):
    low_cut = np.percentile(y_train, 33)
    high_cut = np.percentile(y_train, 67)

    def to_category(values):
        return pd.cut(
            values,
            bins=[-np.inf, low_cut, high_cut, np.inf],
            labels=["Low", "Medium", "High"]
        ).astype(str)

    return to_category, low_cut, high_cut