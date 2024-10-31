from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from ..load_data import read_tables
from ..preprocess_data import preprocess, fit_scale_data, inverse_transform_target
from ..evaluate_model import evaluate_model, plot_curve
from ..save_model import save_model


if __name__ == "__main__":
    df = read_tables()
    tidy_df, X, y = preprocess(df)

    X_scaled, y_scaled = fit_scale_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred_test = model.predict(X_test_poly)
    y_pred = model.predict(X_poly)

    y_test_rescaled, y_pred_test_rescaled = inverse_transform_target(y_test), inverse_transform_target(y_pred_test)
    y_pred_rescaled = inverse_transform_target(y_pred)

    evaluate_model(y_test_rescaled, y_pred_test_rescaled, "Polynomial_Regression", coef=model.coef_, bias=model.intercept_)
    plot_curve(tidy_df, y, y_pred_rescaled.reshape(-1,), "Polynomial_Regression")

    save_model(model, "Polynomial_Regression")