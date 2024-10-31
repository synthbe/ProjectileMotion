import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, r2_score
import os
from typing import TypeAlias
from numpy import typing, float64
from pandas import DataFrame

FloatArray: TypeAlias = typing.NDArray[float64]

results_dir = "results"

def evaluate_model(y_test: FloatArray, y_pred: FloatArray, model_name: str, **kwargs) -> None:
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    file = f"{results_dir}/metrics/{model_name}.txt"
    os.makedirs(results_dir, exist_ok=True)

    with open(file, 'w') as f:
        f.write(f"Root Mean Squared Error: {rmse:.2f} | R2: {r2:.2f}\n")
        [f.write(f"{key} -> {value}\n") for key, value in kwargs.items()]

def plot_curve(data: DataFrame, y_true: FloatArray, y_pred: FloatArray, model_name: str) -> None:
    sns.lineplot(x=data.H.values, y=y_pred, label=model_name)
    sns.scatterplot(x=data.H.values, y=y_true, label="Valores Reais")

    img_dir = f"{results_dir}/images/{model_name}.png"
    os.makedirs(results_dir, exist_ok=True)

    plt.savefig(img_dir)