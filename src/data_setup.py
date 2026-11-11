import pandas as pd
from sklearn.model_selection import train_test_split

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

    before = len(df)
    df = df.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print("Duplicate rows removed:", before - len(df))

    return df


def get_train_test_split(df, test_size=0.2, random_state=42):
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(X,y,test_size=test_size,random_state=random_state,)