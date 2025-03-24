import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Uses Mean Squared Error to calculate the scores for the model
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            None
        """
        try:
            logging.info("Calculating Mean Squared Error (MSE)")
            mse = np.mean((y_true - y_pred) ** 2)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Uses R2 Score to calculate the scores for the model
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            None
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Uses Root Mean Squared Error to calculate the scores for the model
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            None
        """
        try:
            logging.info("Calculating Root Mean Squared Error (RMSE)")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
