import skops.io as sio
import os
from sklearn.base import BaseEstimator

results_dir = "results/models"

def save_model(model: BaseEstimator, model_name: str) -> None:
    file = f"{results_dir}/{model_name}.skops"
    os.makedirs(results_dir, exist_ok=True)

    sio.dump(model, file)