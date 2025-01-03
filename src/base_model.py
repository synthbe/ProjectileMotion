from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .load_data import read_tables, tidy_and_split_data
from .evaluate_model import evaluate_model, plot_curve
from .save_model import save_model

def execute_pipeline(model: BaseEstimator, model_preprocessing: ColumnTransformer, model_name: str, **kwargs) -> None:
    df = read_tables()
    tidy_df, X, y = tidy_and_split_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    pipe = Pipeline(
        steps=[
            ("preprocessing", model_preprocessing),
            (model_name, model(**kwargs))
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X) # For plotting
    y_pred_test = pipe.predict(X_test)

    model_instance = pipe.named_steps[model_name]
    model_attributes = {attr: getattr(model_instance, attr) for attr in dir(model_instance) if not attr.startswith('_')}

    evaluate_model(y_test, y_pred_test, model_name, **model_attributes)
    plot_curve(tidy_df, y, y_pred.reshape(-1,), model_name)
    save_model(pipe, model_name)
