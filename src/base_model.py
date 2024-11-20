from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from load_data import read_tables
from preprocess_data import Preprocesser
from evaluate_model import evaluate_model, plot_curve
from save_model import save_model

preprocesser = Preprocesser(StandardScaler)

def execute_pipeline(model_pipeline, model_name: str) -> None:
    df = read_tables()
    tidy_df, X, y = preprocesser.preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_scaled, y_train_scaled = preprocesser.fit_scale_data(X_train, y_train)

    model_pipeline.fit(X_train_scaled, y_train_scaled)

    X_scaled = preprocesser.transform_features(X)
    X_test_scaled = preprocesser.transform_features(X_test)

    y_pred = model_pipeline.predict(X_scaled)
    y_pred_test = model_pipeline.predict(X_test_scaled)

    y_pred_rescaled = preprocesser.inverse_transform_target(y_pred)
    y_pred_rescaled_test = preprocesser.inverse_transform_target(y_pred_test)

    evaluate_model(y_test, y_pred_rescaled_test, model_name)
    plot_curve(tidy_df, y, y_pred_rescaled.reshape(-1,), model_name)
    save_model(model_pipeline, model_name)
