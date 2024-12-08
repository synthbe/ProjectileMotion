import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, r2_score
import os
from typing import TypeAlias
from numpy import typing, float64
from pandas import DataFrame

FloatArray: TypeAlias = typing.NDArray[float64]

RESULTS_DIR = "results"

def evaluate_model(y_test: FloatArray, y_pred: FloatArray, model_name: str, **kwargs) -> None:
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    file = f"{RESULTS_DIR}/metrics/{model_name}.txt"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(file, 'w') as f:
        f.write("Metrics:\n")
        f.write(f"\tRoot Mean Squared Error: {rmse:.2f} | R2: {r2:.2f}\n")
        f.write("Model attributes\n")
        [f.write(f"\t{key} -> {value}\n") for key, value in kwargs.items()]

def plot_curve(data: DataFrame, y_true: FloatArray, y_pred: FloatArray, model_name: str) -> None:
    sns.lineplot(x=data.H.values, y=y_pred, label=model_name)
    sns.scatterplot(x=data.H.values, y=y_true, label="Valores Reais")

    img_dir = f"{RESULTS_DIR}/images/{model_name}.png"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.savefig(img_dir)
