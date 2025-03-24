import logging
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd

from typing import Union

from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    # This is just an abstract class like a blueprint of how the data should be handled
    # We have to overwrite this to build our own custom solutions.
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

# Strategy 1: Preprocess the data
class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing the data
    """
    # Overwriting the handle_data method
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        """
        try:
            # Droping columns which are categorical (only for learning)
            columns = ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'order_purchase_timestamp']
            data = data.drop(columns, axis=1)
            # We do some EDA to fill the NaN values in certain columns
            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace=True)
            data['review_comment_message'].fillna('No review', inplace=True)
            # Dropping columns which are not numbers (only for learning)
            data = data.select_dtypes(include=[np.number])
            # Dropping columns which are not useful (only for learning)
            columns_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(columns_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

# Strategy 2: Divide the data into train and test        
class DataDivideStrategy(DataStrategy):
    """
    Divide the data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # train = pd.concat([X_train, y_train], axis=1)
            # test = pd.concat([X_test, y_test], axis=1)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

# To Run both the strategies
class DataCleaning:
    """
    Data Cleaning class which transforms the data into clean data.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.df = data
        self.strategy = strategy
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

# if __name__ == "__main__":
#     df = pd.read_csv("data/olist_customers_dataset.csv")
#     strategy = DataCleaning(df,DataPreProcessStrategy())
#     data_cleaning = DataCleaning(df, strategy)
#     data_cleaning.handle_data()