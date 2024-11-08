from typing import Tuple, TypeAlias
from sklearn.base import TransformerMixin
from pandas import DataFrame
from numpy import typing, float64

FloatArray: TypeAlias = typing.NDArray[float64]

class Preprocesser:

  def __init__(self, Scaler: TransformerMixin) -> None:
    self.scaler_X = Scaler()
    self.scaler_y = Scaler()


  def preprocess_data(self, df: DataFrame) -> Tuple[DataFrame, FloatArray, FloatArray]:
    """
    Returns the dataframe as tidy pattern and splited
    """

    tidy_df: DataFrame = df.melt(id_vars=['h'], var_name='H', value_name='R')
    tidy_df: DataFrame = tidy_df.map(float).sort_values(by='H')
    tidy_df.dropna(inplace=True)

    X = tidy_df.drop(columns=['R'])
    y = tidy_df.R

    return tidy_df, X.values, y.values


  def fit_scale_data(self, X: FloatArray, y: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """
    Returns X and y scaled transformed
    """

    X_scaled = self.scaler_X.fit_transform(X) 
    y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)) 

    return X_scaled, y_scaled

  def inverse_transform_data(self, X_transformed: FloatArray, y_transformed: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """
    Returns X and y back to the original scale
    """

    X_rescaled = self.scaler_X.inverse_transform(X_transformed)
    y_rescaled = self.scaler_y.inverse_transform(y_transformed)

    return X_rescaled, y_rescaled

  def transform_features(self, X: FloatArray) -> FloatArray:
    """
    Returns just X scaled transformed
    """
    return self.scaler_X.transform(X)

  def transform_target(self, y: FloatArray) -> FloatArray:
    """
    Returns just y scaled transformed
    """
    return self.scaler_y.transform(y)

  def inverse_transform_features(self, X_transformed: FloatArray) -> FloatArray:
    """
    Returns just X back to the original scale
    """

    return self.scaler_X.inverse_transform(X_transformed)

  def inverse_transform_target(self, y_transformed: FloatArray) -> FloatArray:
    """
    Returns just y back to the original scale
    """

    return self.scaler_y.inverse_transform(y_transformed)