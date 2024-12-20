# Function: Transform Dates

import pandas as pd
import numpy as np

def transform_dates(dataframe, date_column, granularity='monthly'):
    """
    Transforms a date column into features suitable for machine learning models.
    """
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    dataframe['year'] = dataframe[date_column].dt.year

    if granularity == 'monthly':
        dataframe['month'] = dataframe[date_column].dt.month
        dataframe['month_sin'] = np.sin(2 * np.pi * dataframe['month'] / 12)
        dataframe['month_cos'] = np.cos(2 * np.pi * dataframe['month'] / 12)
    elif granularity == 'weekly':
        dataframe['week'] = dataframe[date_column].dt.isocalendar().week
        dataframe['week_sin'] = np.sin(2 * np.pi * dataframe['week'] / 52)
        dataframe['week_cos'] = np.cos(2 * np.pi * dataframe['week'] / 52)
    elif granularity == 'daily':
        dataframe['day'] = dataframe[date_column].dt.day
        dataframe['day_sin'] = np.sin(2 * np.pi * dataframe['day'] / 31)
        dataframe['day_cos'] = np.cos(2 * np.pi * dataframe['day'] / 31)

    dataframe = dataframe.drop(columns=[date_column])  # Drop the original date column
    return dataframe