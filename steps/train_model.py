import logging

import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Trains the Model on ingested Data
    Args:
       df: pd.DataFrame
    """
    pass