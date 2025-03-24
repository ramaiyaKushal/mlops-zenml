import logging

import pandas as pd
from zenml import step
from src.model_development import LinearRegressionModel, RandomForestModel
from sklearn.base import RegressorMixin

import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
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
         mlflow.sklearn.autolog()  # Experiment tracking using mlflow autologging for sklearn models
         model = LinearRegressionModel()
         trained_model = model.train(X_train, y_train)
         return trained_model
      if model_name == "RandomForest":
         model = RandomForestModel()
         trained_model = model.train(X_train, y_train)
         return trained_model
      else:
         raise ValueError("Model {} not supported".format(model_name))
   except Exception as e:
      logging.error("Error in training model: {}".format(e))
      raise e