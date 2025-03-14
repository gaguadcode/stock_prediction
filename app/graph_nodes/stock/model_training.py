import pickle
import redis
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from mlflow.models import infer_signature
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState

class StockDataTrainer:
    """
    Trains a Gradient Boosting model using stock data, logs training 
    results in MLflow, and saves the model in Redis.
    """

    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0, date_column='date', target_column='price'):
        self.logger = get_logger(self.__class__.__name__)
        self.date_column = date_column
        self.target_column = target_column
        self.model = None
        self.mse = None

        # ✅ MLflow Setup
        mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Ensure MLflow is running
        mlflow.set_experiment("Stock Prediction Training")

        # ✅ Redis Connection
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

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

    def train_model(self, X_train, y_train):
        """
        Trains a Gradient Boosting regression model.
        """
        self.logger.info("Training Gradient Boosting model...")
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.model.fit(X_train, y_train)
        self.logger.info("Model training complete.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model and stores the Mean Squared Error (MSE).
        """
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.logger.info(f"Model Evaluation Complete - MSE: {self.mse}")

    def save_model_to_mlflow(self, X_sample):
        """
        Logs the trained model and evaluation metrics in MLflow with input example and signature.
        """
        with mlflow.start_run():
            mlflow.log_param("model_type", "GradientBoostingRegressor")
            mlflow.log_metric("mse", self.mse)

            # ✅ Generate signature
            signature = infer_signature(X_sample, self.model.predict(X_sample))

            # ✅ Log model with signature and input example
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature,
                input_example=X_sample[:5]  # Provide a small input sample
            )

            self.logger.info("Model logged to MLflow with signature and input example.")

    def save_model_to_redis(self, model_key="trained_model"):
        """
        Serializes and stores the trained model in Redis.
        """
        if self.model is None:
            self.logger.error("No trained model found to save.")
            return

        model_bytes = pickle.dumps(self.model)
        self.redis_client.set(model_key, model_bytes)
        self.logger.info(f"Model saved to Redis with key: {model_key}")

    def execute_training(self, state: WorkflowState) -> WorkflowState:
        """
        Fetches data, preprocesses, trains the model, logs to MLflow, saves it in Redis, 
        and updates only `None` values in the WorkflowState.
        """
        if not state.database_url:
            raise ValueError("❌ Database URL is missing. Ensure data has been fetched before training.")

        granularity = self.get_granularity(state.date_period)
        self.logger.info(f"📊 Using granularity: {granularity}")

        # ✅ Fetch and preprocess data
        df = self.fetch_data_from_db(state.database_url)
        X, y = self.preprocess_data(df, granularity)

        # ✅ Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ✅ Train and Evaluate
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)

        # ✅ Save Model
        self.save_model_to_mlflow(X_train)  # Log in MLflow
        self.save_model_to_redis()  # Save in Redis

        # ✅ Prepare updated data while preserving existing values
        updated_data = {
            "mse": self.mse if state.mse is None else state.mse  # Update only if `mse` is None
        }

        # ✅ Merge updated data while preserving state
        return state.model_copy(update=updated_data)
