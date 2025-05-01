import os
import polars as pl
from config.settings import RAW_DATA_DIR, RAW_DATA_FILE


def load_dataset(dataset_name: str = RAW_DATA_FILE) -> pl.DataFrame:
    """
    Load the dataset from the csv file

    Args:
        dataset_name (str): The name of the dataset to load

    Returns:
        pl.DataFrame: The loaded dataset containing data gathered by langchain
    """

    file_path = os.path.join(RAW_DATA_DIR, dataset_name)
    df = pl.read_csv(file_path)
    return df
