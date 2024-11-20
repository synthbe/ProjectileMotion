from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from ..base_model import execute_pipeline

cols = [0, 1]
linear_regression_preprocessing = ColumnTransformer([("scaler", StandardScaler(), cols)])

if __name__ == "__main__":
    execute_pipeline(LinearRegression, linear_regression_preprocessing, "Linear_Regression_pipe")