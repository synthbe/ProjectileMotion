from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

from ..load_data import read_tables
from ..preprocess_data import Preprocesser
from ..evaluate_model import evaluate_model, plot_curve
from ..save_model import save_model

preprocesser = Preprocesser(StandardScaler)


if __name__ == "__main__":
    df = read_tables()
    tidy_df, X, y = preprocesser.preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_scaled, y_train_scaled = preprocesser.fit_scale_data(X_train, y_train)

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_train_scaled_poly = poly.fit_transform(X_train_scaled)
    

    model = LinearRegression()
    model.fit(X_train_scaled_poly, y_train_scaled)
    
    X_scaled = preprocesser.transform_features(X)
    X_test_scaled = preprocesser.transform_features(X_test)
    X_scaled_poly = poly.transform(X_scaled)
    X_test_scaled_poly = poly.transform(X_test_scaled)

    y_pred_test = model.predict(X_test_scaled_poly)
    y_pred = model.predict(X_scaled_poly) # Predictions from the whole data to be ploted

    y_pred_test_rescaled = preprocesser.inverse_transform_target(y_pred_test)
    y_pred_rescaled = preprocesser.inverse_transform_target(y_pred)

    evaluate_model(y_test, y_pred_test_rescaled, "Polynomial_Regression", coef=model.coef_, bias=model.intercept_)
    plot_curve(tidy_df, y, y_pred_rescaled.reshape(-1,), "Polynomial_Regression")

    save_model(model, "Polynomial_Regression")