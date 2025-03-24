import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract class for all models.
    """
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the model on the given data.
        """
        pass
        # @abstractmethod
    # def predict(self, X_test: pd.DataFrame) -> pd.Series:
    #     """
    #     Predicts the labels for the given data.
    #     """
    #     pass
    # @abstractmethod
    # def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    #     """
    #     Evaluates the model on the given data.
    #     """
    #     pass
    # @abstractmethod

class LinearRegressionModel(Model):
    """
    Linear Regression Model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,**kwargs) -> None:
        """
        Trains the model on the given data.
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
    
    # def predict(self, X_test: pd.DataFrame) -> pd.Series:
    #     """
    #     Predicts the labels for the given data.
    #     """
    #     pass
    # def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    #     """
    #     Evaluates the model on the given data.
    #     """
    #     pass

class RandomForestModel(Model):
    """
    Random Forest Model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,**kwargs) -> None:
        """
        Trains the model on the given data.
        """
        try:
            reg = RandomForestRegressor(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
        
