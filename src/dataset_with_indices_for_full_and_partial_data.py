import pandas as pd
from pandas.api.types import is_numeric_dtype
from dataclasses import dataclass, field
from types import FunctionType


def encode_categorical_features(df):
    categorical_columns = []
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            categorical_columns.append(c)

    for c in categorical_columns:
        one_hot = pd.get_dummies(df[c], prefix=c, dtype=int)
        one_hot.columns = [c.replace(' ', '_').replace(',', '-').replace('<', '(').replace('>', ')') for c in
                           one_hot.columns]
        df = df.join(one_hot).drop(columns=c)
    return df


def null_columns(df):
    for c in df.columns:
        if any(df[c].isna().tolist()):
            df[c + '_null'] = df[c].isna().astype(int)
    df.fillna(0, inplace=True)
    return df


@dataclass
class Index_Dataset:
    df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    target_col: str
    metric: FunctionType
    index: pd.Index = field(init=False)  # Add index as a field that is initialized later

    def __post_init__(self):
        self.assign_index()

    def assign_index(self):
        """Assign a default or custom index to the dataset."""
        # You can customize how the index is assigned
        if self.df.index.isnull().any():
            self.df.index = pd.Index([i for i in range(len(self.df))], name="index")
        self.index = self.df.index  # Store the index in the dataset property

    def add_missing_features(self, reference_columns):
        """Add missing features to the dataset based on reference columns."""
        # Identify missing columns
        missing_cols = set(reference_columns) - set(self.X_train.columns)
        for col in missing_cols:
            # Add missing columns with default value 0
            self.X_train[col] = 0
            self.X_test[col] = 0
        # Reorder columns to match the reference columns
        self.X_train = self.X_train.loc[:, reference_columns]
        self.X_test = self.X_test.loc[:, reference_columns]