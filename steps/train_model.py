import logging

import pandas as pd
from zenml import step
from src.model_development import LinearRegressionModel, RandomForestModel
from sklearn.base import RegressorMixin

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "LinearRegression") -> RegressorMixin:
   """
   Trains the Model on ingested Data
   Args:
      X_train: pd.DataFrame
      y_train: pd.Series
   Returns:
      model: RegressorMixin
   """
   try:
      model = None
      if model_name == "LinearRegression":
         model = LinearRegressionModel()
         trained_model = model.train(X_train, y_train)
         return trained_model
      if model_name == "RandomForest":
         model = RandomForestModel()
         trained_model = model.train(X_train, y_train)
         return trained_model
      else:
         raise ValueError("Model {} not supported".format(config.model_name))
   except Exception as e:
      logging.error("Error in training model: {}".format(e))
      raise e