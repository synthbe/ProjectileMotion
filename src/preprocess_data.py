from sklearn.preprocessing import StandardScaler
from typing import Tuple, TypeAlias
from pandas import DataFrame
from numpy import typing, float64

FloatArray: TypeAlias = typing.NDArray[float64]

scaler_X = StandardScaler()
scaler_y = StandardScaler()


def preprocess(df: DataFrame) -> Tuple[DataFrame, FloatArray, FloatArray]:
  """
  Returns the dataframe as tidy pattern and splited
  """

  tidy_df: DataFrame = df.melt(id_vars=['h'], var_name='H', value_name='R')
  tidy_df: DataFrame = tidy_df.map(float).sort_values(by='H')
  tidy_df.dropna(inplace=True)

  X = tidy_df.drop(columns=['R'])
  y = tidy_df.R

  return tidy_df, X.values, y.values


def fit_scale_data(X: FloatArray, y: FloatArray) -> Tuple[FloatArray, FloatArray]:
  X_scaled = scaler_X.fit_transform(X) 
  y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)) 

  return X_scaled, y_scaled

def transform_features(X: FloatArray) -> FloatArray:
  return scaler_X.transform(X)

def transform_target(y: FloatArray) -> FloatArray:
  return scaler_y.transform(y)

def inverse_transform_features(X_transformed: FloatArray) -> FloatArray:
  return scaler_X.inverse_transform(X_transformed)

def inverse_transform_target(y_transformed: FloatArray) -> FloatArray:
  return scaler_y.inverse_transform(y_transformed)