import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from app.utils.config import config

class GradientBoostingWorkflow:
    def __init__(self, granularity='monthly', date_column='date', target_column='price'):
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
            print("\n[DEBUG] Loading data from CSV...")
            dataframe = pd.read_csv(self.data_path)
            if dataframe.empty:
                raise ValueError("The loaded CSV file is empty.")
            print("\n[DEBUG] Data loaded successfully. First few rows:")
            print(dataframe.head())  # Debug: Check the first few rows
            self.data = dataframe
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at the path: {self.data_path}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the CSV file: {e}")

    def transform_dates(self, dataframe):
        """
        Transforms the date column into machine-learning-friendly features.
        """
        try:
            # Ensure the date_column exists
            if self.date_column not in dataframe.columns:
                raise ValueError(f"Date column '{self.date_column}' is missing from the dataframe.")

            # Convert the date column to datetime
            dataframe[self.date_column] = pd.to_datetime(dataframe[self.date_column])
            dataframe['year'] = dataframe[self.date_column].dt.year

            # Add features based on the specified granularity
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

            # Drop the original date column
            dataframe = dataframe.drop(columns=[self.date_column])
            return dataframe

        except Exception as e:
            raise Exception(f"An error occurred in transform_dates: {e}")

    def preprocess_data(self):
        """
        Prepares the features (X) and target (y) for training.
        """
        print("\n[DEBUG] Preprocessing data...")
        data_transformed = self.transform_dates(self.data)
        X = data_transformed.drop(columns=[self.target_column])
        y = data_transformed[self.target_column]
        print("\n[DEBUG] Preprocessed data:")
        print("Features (X):")
        print(X.head())
        print("Target (y):")
        print(y.head())
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        print("\n[DEBUG] Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Trains a Gradient Boosting regression model.
        """
        print("\n[DEBUG] Training the Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.model.fit(X_train, y_train)
        print("\n[DEBUG] Model training complete.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model using Mean Squared Error.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        print("\n[DEBUG] Evaluating the model...")
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {self.mse}")

    def make_predictions(self, new_data):
        """
        Transforms new data and makes predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        print("\n[DEBUG] Making predictions for new data...")
        new_data_transformed = self.transform_dates(new_data)
        predictions = self.model.predict(new_data_transformed)
        print("\n[DEBUG] Predictions:")
        print(predictions[:5])  # Show the first few predictions
        return predictions

    def execute_workflow(self, new_data):
        """
        Executes the full workflow: Preprocessing, splitting, training, evaluation, and prediction.

        Args:
            new_data (pd.DataFrame): New data for prediction.

        Returns:
            tuple: (mse, predictions)
        """
        print("\n[DEBUG] Starting workflow execution...")
        self.load_data()
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        predictions = self.make_predictions(new_data)
        print("\n[DEBUG] Workflow execution complete.")
        return self.mse, predictions
