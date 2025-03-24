import logging

import pandas as pd
from zenml import step
from src.model_development import LinearRegressionModel, RandomForestModel
from sklearn.base import RegressorMixin
from .config import ModelnameConfig

@step
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,config: ModelnameConfig) -> RegressorMixin:
   """
   Trains the Model on ingested Data
   Args:
      X_train: pd.DataFrame
      X_test: pd.DataFrame
      y_train: pd.Series
      y_test: pd.Series
   Returns:
      model: RegressorMixin
   """
   try:
      model = None
      if config.model_name == "LinearRegression":
         model = LinearRegressionModel()
         trained_model = model.train(X_train, y_train)
         return trained_model
      if config.model_name == "RandomForest":
         model = RandomForestModel()
         trained_model = model.train(X_train, y_train)
         return trained_model
      else:
         raise ValueError("Model {} not supported".format(config.model_name))
   except Exception as e:
      logging.error("Error in training model: {}".format(e))
      raise e