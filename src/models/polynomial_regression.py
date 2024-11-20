from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..base_model import execute_pipeline

cols = [0, 1]
polynomial_regression_preprocessing = ColumnTransformer([
    ("scaler", StandardScaler(), cols),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=True), cols)
])

if __name__ == "__main__":
    execute_pipeline(LinearRegression, polynomial_regression_preprocessing, "Polynomial_Regression_pipe")