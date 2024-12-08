from sklearn.neural_network import MLPRegressor

from ..base_model import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        MLPRegressor, None, "Multilayer_Perceptron",
        hidden_layer_sizes=(16, 32),
        max_iter=400,
        random_state=42,
    )