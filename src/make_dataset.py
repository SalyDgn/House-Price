from typing import Optional
import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_openml


def load_data(dataset_name, column_to_lower: Optional[bool] = False) -> pd.DataFrame:
  """Fetch dataset from OpenML by name.

  Args:
    dataset_name (str): dataset name to load
    column_to_lower (Optionnal[bool]): default is Trus, flag to know if we transform colums names to lower

  Returns:
    pd.DataFrame: feature and target data
  """
  if not dataset_name:
    raise ValueError("Dataset name is required")
  logger.info(f"Dataset to load : {dataset_name}")
  if dataset_name == 'house_prices':
    dframe_house= fetch_openml(dataset_name, return_X_y=False, target_column=None, parser='auto')
    data = dframe_house.data

  if column_to_lower:
      logger.info(f'Columns will be transformed to lower')
      data.columns = data.columns.str.lower()

  logger.info(f'Data shape: {data.shape}')
  logger.info(f'Data description: {dframe_house.DESCR}')

  return data
  
