import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from config import config

class GradientBoostingWorkflow:
    def __init__(self, granularity='monthly', date_column='month', target_column='price'):
        """
        Initialize the workflow with configuration options.

        Args:
            granularity (str): Granularity of the data ('monthly', 'weekly', 'daily').
            date_column (str): Name of the date column.
            target_column (str): Name of the target column.
        """
        self.granularity = granularity
        self.date_column = date_column
        self.target_column = target_column
        self.data  = config.OUTPUT_CSV
        self.model = None
        self.mse = None

    def transform_dates(self, dataframe):
        """
        Transforms the date column into machine-learning-friendly features.
        """
        dataframe[self.date_column] = pd.to_datetime(dataframe[self.date_column])
        dataframe['year'] = dataframe[self.date_column].dt.year

        if self.granularity == 'monthly':
            dataframe['month'] = dataframe[self.date_column].dt.month
            dataframe['month_sin'] = np.sin(2 * np.pi * dataframe['month'] / 12)
            dataframe['month_cos'] = np.cos(2 * np.pi * dataframe['month'] / 12)
        elif self.granularity == 'weekly':
            dataframe['week'] = dataframe[self.date_column].dt.isocalendar().week
            dataframe['week_sin'] = np.sin(2 * np.pi * dataframe['week'] / 52)
            dataframe['week_cos'] = np.cos(2 * np.pi * dataframe['week'] / 52)
        elif self.granularity == 'daily':
            dataframe['day'] = dataframe[self.date_column].dt.day
            dataframe['day_sin'] = np.sin(2 * np.pi * dataframe['day'] / 31)
            dataframe['day_cos'] = np.cos(2 * np.pi * dataframe['day'] / 31)

        dataframe = dataframe.drop(columns=[self.date_column])  # Drop the original date column
        return dataframe

    def preprocess_data(self, data):
        """
        Prepares the features (X) and target (y) for training.
        """
        data_transformed = self.transform_dates(data)
        X = data_transformed.drop(columns=[self.target_column])
        y = data_transformed[self.target_column]
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Trains a Gradient Boosting regression model.
        """
        self.model = GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model using Mean Squared Error.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {self.mse}")

    def make_predictions(self, new_data):
        """
        Transforms new data and makes predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        new_data_transformed = self.transform_dates(new_data)
        predictions = self.model.predict(new_data_transformed)
        return predictions

    def execute_workflow(self, new_data):
        """
        Executes the full workflow: Preprocessing, splitting, training, evaluation, and prediction.

        Args:
            data (pd.DataFrame): Training data.
            new_data (pd.DataFrame): New data for prediction.

        Returns:
            tuple: (mse, predictions)
        """
        # Preprocess data
        X, y = self.preprocess_data(self.data)
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        # Train model
        self.train_model(X_train, y_train)
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        # Make predictions
        predictions = self.make_predictions(new_data)
        return self.mse, predictions

