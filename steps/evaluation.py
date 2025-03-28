import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame, 
                   y_test: pd.Series) -> Tuple[
                       Annotated[float, "mse"], 
                       Annotated[float, "r2_score"], 
                       Annotated[float, "rmse"]
                       ]:
    """
    Evaluates the model on the ingested data
    Args:
        pd.DataFrame : DataFrame containing the data
    Returns:
        None
    """
    try:
        logging.info("Evaluating model")
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        return mse, r2_score, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e