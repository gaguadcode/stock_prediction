import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState

class StockDataTrainer:
    """
    Trains a Gradient Boosting model using stock data, 
    performs hyperparameter tuning, logs training results in MLflow,
    and updates the 'latest' alias when a better model is found.
    """

    def __init__(self, date_column='date', target_column='price'):
        self.logger = get_logger(self.__class__.__name__)
        self.date_column = date_column
        self.target_column = target_column
        self.model = None
        self.mse = None
        self.best_params = None  # Stores best hyperparameters
        self.model_name = "GradientBoostingRegressor"
        self.mlflow_client = MlflowClient()

        # ✅ MLflow Setup
        mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Ensure MLflow is running
        mlflow.set_experiment("Stock Prediction Training")

    def get_granularity(self, date_period: str) -> str:
        """
        Determines granularity based on the date_period in WorkflowState.
        """
        period_mapping = {
            "TIME_SERIES_DAILY": "daily",
            "TIME_SERIES_WEEKLY": "weekly",
            "TIME_SERIES_MONTHLY": "monthly"
        }
        return period_mapping.get(date_period, "monthly")  # Default to 'monthly'

    def fetch_data_from_db(self, database_url: str) -> pd.DataFrame:
        """
        Fetch stock data from the database using the provided database URL.
        """
        self.logger.info(f"Connecting to database: {database_url}")
        engine = create_engine(database_url)
        query = "SELECT * FROM historical_stock_data"

        try:
            df = pd.read_sql(query, engine)
            if df.empty:
                raise ValueError("No data retrieved from database.")
            self.logger.info(f"Fetched {len(df)} rows from database.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from database: {e}")
            raise

    def transform_dates(self, dataframe: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """
        Transforms the date column into machine-learning-friendly features.
        """
        try:
            self.logger.info(f"Transforming date column for granularity: {granularity}")
            dataframe[self.date_column] = pd.to_datetime(dataframe[self.date_column])
            dataframe['year'] = dataframe[self.date_column].dt.year

            if granularity == 'monthly':
                dataframe['month'] = dataframe[self.date_column].dt.month
                dataframe['month_sin'] = np.sin(2 * np.pi * dataframe['month'] / 12)
                dataframe['month_cos'] = np.cos(2 * np.pi * dataframe['month'] / 12)
            elif granularity == 'weekly':
                dataframe['week'] = dataframe[self.date_column].dt.isocalendar().week
                dataframe['week_sin'] = np.sin(2 * np.pi * dataframe['week'] / 52)
                dataframe['week_cos'] = np.cos(2 * np.pi * dataframe['week'] / 52)
            elif granularity == 'daily':
                dataframe['day'] = dataframe[self.date_column].dt.day
                dataframe['day_sin'] = np.sin(2 * np.pi * dataframe['day'] / 31)
                dataframe['day_cos'] = np.cos(2 * np.pi * dataframe['day'] / 31)

            dataframe.drop(columns=[self.date_column], inplace=True)
            return dataframe
        except Exception as e:
            self.logger.error(f"Error in transform_dates: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame, granularity: str):
        """
        Prepares the features (X) and target (y) for training.
        """
        self.logger.info("Preprocessing data...")

        # ✅ Transform date features
        df_transformed = self.transform_dates(df, granularity)

        # ✅ Extract features (X) and target (y)
        X = df_transformed.drop(columns=[self.target_column], errors="ignore").select_dtypes(include=[np.number])
        y = df_transformed[self.target_column]

        return X, y

    def train_model_with_hyperparameter_tuning(self, X_train, y_train):
        """
        Performs randomized hyperparameter tuning for Gradient Boosting Regressor.
        """
        self.logger.info("🔍 Performing hyperparameter tuning...")

        # ✅ Define parameter grid
        param_dist = {
            "n_estimators": np.arange(50, 300, 50),
            "learning_rate": np.linspace(0.01, 0.3, 10),
            "max_depth": np.arange(3, 10),
            "subsample": np.linspace(0.6, 1.0, 5),
            "min_samples_split": np.arange(2, 10),
            "min_samples_leaf": np.arange(1, 10)
        }

        # ✅ Create base model
        model = GradientBoostingRegressor(random_state=42)

        # ✅ Perform randomized search
        search = RandomizedSearchCV(
            model, param_distributions=param_dist, 
            n_iter=20, cv=3, scoring='neg_mean_squared_error', 
            verbose=2, random_state=42, n_jobs=-1
        )

        search.fit(X_train, y_train)

        # ✅ Store best model and parameters
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.logger.info(f"✅ Best hyperparameters found: {self.best_params}")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model and stores the Mean Squared Error (MSE).
        """
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.logger.info(f"✅ Model Evaluation Complete - MSE: {self.mse}")

    def register_and_update_alias(self, run_id):
        """
        Registers the model and updates the 'latest' alias if performance improves.
        """
        model_uri = f"runs:/{run_id}/model"
        client = self.mlflow_client

        # ✅ Ensure the model is registered before adding versions
        try:
            client.get_registered_model(self.model_name)
            self.logger.info(f"✅ Registered model '{self.model_name}' already exists.")
        except mlflow.exceptions.RestException:
            self.logger.info(f"⚠️ Registered model '{self.model_name}' not found. Creating new model in registry...")
            client.create_registered_model(self.model_name)

        # ✅ Register new model version
        model_version = client.create_model_version(
            name=self.model_name, source=model_uri, run_id=run_id
        )

        # ✅ Check if there's a current 'latest' model
        try:
            latest_version = client.get_model_version_by_alias(self.model_name, "latest")
            latest_version_mse = float(latest_version.tags.get("mse", float('inf')))
        except Exception:
            latest_version = None
            latest_version_mse = float('inf')

        # ✅ Compare MSE and update alias if the new model is better
        if self.mse < latest_version_mse:
            client.set_model_version_tag(model_version.name, model_version.version, "mse", str(self.mse))
            client.set_registered_model_alias(self.model_name, "latest", model_version.version)
            self.logger.info(f"✅ Updated 'latest' alias to version {model_version.version} with MSE {self.mse}")


    def execute_training(self, state: WorkflowState) -> WorkflowState:
        """
        Fetches data, preprocesses, trains the model with hyperparameter tuning, 
        logs to MLflow, and updates only `None` values in the WorkflowState.
        """
        df = self.fetch_data_from_db(state.database_url)
        X, y = self.preprocess_data(df, self.get_granularity(state.date_period))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_model_with_hyperparameter_tuning(X_train, y_train)
        self.evaluate_model(X_test, y_test)

        with mlflow.start_run() as run:
            self.register_and_update_alias(run.info.run_id)

        return state.model_copy(update={"mse": self.mse})
