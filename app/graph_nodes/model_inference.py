import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from app.utils.config import config
from app.utils.logger import get_logger

class GradientBoostingWorkflow:
    def __init__(self, granularity='monthly', date_column='date', target_column='price'):
        self.logger = get_logger(self.__class__.__name__)
        self.granularity = granularity
        self.date_column = date_column
        self.target_column = target_column
        self.data_path = config.OUTPUT_CSV  # Path to the CSV file
        self.data = None  # Placeholder for loaded DataFrame
        self.model = None
        self.mse = None

    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame.
        """
        try:
            self.logger.info("Loading data from CSV...")
            dataframe = pd.read_csv(self.data_path)
            if dataframe.empty:
                raise ValueError("The loaded CSV file is empty.")
            self.logger.info("Data loaded successfully. First few rows:")
            self.logger.debug("\n%s", dataframe.head())  # Debug-level logging for data
            self.data = dataframe
        except FileNotFoundError:
            self.logger.error("CSV file not found at the path: %s", self.data_path)
            raise
        except Exception as e:
            self.logger.error("An error occurred while loading the CSV file: %s", e)
            raise

    def transform_dates(self, dataframe):
        """
        Transforms the date column into machine-learning-friendly features.
        """
        try:
            self.logger.info("Transforming date column into features...")
            if self.date_column not in dataframe.columns:
                raise ValueError(f"Date column '{self.date_column}' is missing from the dataframe.")

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

            dataframe = dataframe.drop(columns=[self.date_column])
            self.logger.info("Date transformation complete.")
            return dataframe
        except Exception as e:
            self.logger.error("An error occurred in transform_dates: %s", e)
            raise

    def preprocess_data(self):
        """
        Prepares the features (X) and target (y) for training.
        """
        self.logger.info("Preprocessing data...")
        data_transformed = self.transform_dates(self.data)
        X = data_transformed.drop(columns=[self.target_column])
        y = data_transformed[self.target_column]
        self.logger.debug("Features (X):\n%s", X.head())
        self.logger.debug("Target (y):\n%s", y.head())
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        self.logger.info("Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Trains a Gradient Boosting regression model.
        """
        self.logger.info("Training the Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.model.fit(X_train, y_train)
        self.logger.info("Model training complete.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model using Mean Squared Error.
        """
        if self.model is None:
            self.logger.error("Model is not trained yet!")
            raise ValueError("Model is not trained yet!")
        self.logger.info("Evaluating the model...")
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.logger.info("Mean Squared Error: %.4f", self.mse)

    def make_predictions(self, new_data):
        """
        Transforms new data and makes predictions using the trained model.
        """
        if self.model is None:
            self.logger.error("Model is not trained yet!")
            raise ValueError("Model is not trained yet!")
        self.logger.info("Making predictions for new data...")
        new_data_transformed = self.transform_dates(new_data)
        predictions = self.model.predict(new_data_transformed)
        self.logger.debug("Predictions: %s", predictions[:5])  # Show the first few predictions
        return predictions

    def execute_workflow(self, new_data):
        """
        Executes the full workflow: Preprocessing, splitting, training, evaluation, and prediction.

        Args:
            new_data (pd.DataFrame): New data for prediction.

        Returns:
            tuple: (mse, predictions)
        """
        self.logger.info("Starting workflow execution...")
        self.load_data()
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        predictions = self.make_predictions(new_data)
        self.logger.info("Workflow execution complete.")
        return self.mse, predictions