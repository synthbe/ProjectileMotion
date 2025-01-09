import numpy as np
import pandas as pd

def theorical_range(H: float, h: float, rotation: float=0.714285) -> float:
    """
    This function applies the formula to calculate the range of the launch of the sphere
    """
    return 2 * np.sqrt(rotation * (h * (H - h)))

def generate_data(row, epsilon: float=2.3347) -> float:
    """
    This function is intended to generate sythetic data in a rule based method, which is the formula to calculate the range of the sphere in the launch
    """
    if pd.isna(row['R']):
        return theorical_range(H=row['H'], h=row['h']) - epsilon

    return row['R']
