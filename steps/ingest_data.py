import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self,data_path: str) -> None:
        """Initialize the data ingestion class."""
        self.data_path = data_path
        pass

    def get_data(self) -> pd.DataFrame:
        logging.info("Ingesting Data from {self.data_path}")
        return pd.read_csv(self.data_path)
        


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingests data from the source and returns a DataFrame.
    Args:
        str : path to the data
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e
