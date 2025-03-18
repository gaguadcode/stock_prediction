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
from app.utils.utils import anonymize_database_url

class StockDataTrainer:
    """
    Trains a Gradient Boosting model using stock data, 
    performs hyperparameter tuning, logs training results in MLflow,
    and updates the 'best' alias when a better model is found.
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
        database_url_anonimized = anonymize_database_url(database_url)
        self.logger.info(f"📡 Connecting to database: {database_url_anonimized}")
        engine = create_engine(database_url)
        query = "SELECT * FROM historical_stock_data"

        try:
            df = pd.read_sql(query, engine)
            if df.empty:
                raise ValueError("❌ No data retrieved from database.")
            self.logger.info(f"✅ Fetched {len(df)} rows from database.")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error fetching data from database: {e}")
            raise

    def transform_dates(self, dataframe: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """
        Transforms the date column into machine-learning-friendly features.
        """
        try:
            self.logger.info(f"🔄 Transforming date column for granularity: {granularity}")
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
            self.logger.error(f"❌ Error in transform_dates: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame, granularity: str):
        """
        Prepares the features (X) and target (y) for training.
        """
        self.logger.info("🔄 Preprocessing data...")

        # ✅ Apply date transformations
        df_transformed = self.transform_dates(df, granularity)

        # ✅ Extract features (X) and target (y)
        if self.target_column not in df_transformed.columns:
            raise ValueError(f"❌ Target column '{self.target_column}' not found in DataFrame.")

        X = df_transformed.drop(columns=[self.target_column], errors="ignore").select_dtypes(include=[np.number])
        y = df_transformed[self.target_column]

        self.logger.info(f"✅ Data preprocessing complete. Feature shape: {X.shape}, Target shape: {y.shape}")
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

        # ✅ Perform randomized search
        search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42), 
            param_distributions=param_dist, 
            n_iter=20, cv=3, scoring='neg_mean_squared_error', 
            verbose=2, random_state=42, n_jobs=-1
        )

        search.fit(X_train, y_train)

        # ✅ Store best model and parameters
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.logger.info(f"✅ Best hyperparameters found: {self.best_params}")

    def evaluate_and_log_model(self, X_test, y_test):
        """
        Evaluates the model, logs it to MLflow, and updates the 'best' alias if necessary.
        """
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.logger.info(f"✅ Model Evaluation Complete - MSE: {self.mse}")

        with mlflow.start_run() as run:
            # ✅ Log hyperparameters and metrics
            mlflow.log_params(self.best_params)
            mlflow.log_metric("mse", self.mse)

            # ✅ Log the model with input example and signature
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(self.model, artifact_path="model", signature=signature)

            self.register_and_update_alias(run.info.run_id)
            print("este es el runid")
            print(run.info.run_id)

    def register_and_update_alias(self, run_id):
        """
        Registers the model in MLflow and updates the 'best' alias if performance improves.
        Logs extensive details to ensure proper tracking.
        """
        model_uri = f"runs:/{run_id}/model"
        client = self.mlflow_client

        # ✅ Log the model URI before proceeding
        self.logger.info(f"🔍 Model URI for registration: {model_uri}")

        # ✅ Check all registered models before registering
        registered_models = [m.name for m in client.search_registered_models()]
        self.logger.info(f"📜 Currently registered models in MLflow: {registered_models}")

        # ✅ Ensure the model is registered before adding versions
        if self.model_name not in registered_models:
            client.create_registered_model(self.model_name)
            self.logger.info(f"✅ Registered model '{self.model_name}' created successfully.")
        else:
            self.logger.info(f"ℹ️ Model '{self.model_name}' already exists in registry.")

        # ✅ Register new model version
        model_version = client.create_model_version(
            name=self.model_name, source=model_uri, run_id=run_id
        )
        self.logger.info(f"✅ New model version registered: Version {model_version.version}")

        # ✅ Check if there's an existing 'best' alias version
        try:
            best_version = client.get_model_version_by_alias(self.model_name, "best")
            best_version_mse = float(best_version.tags.get("mse", float('inf')))
            self.logger.info(f"🔍 Current 'best' model version: {best_version.version} (MSE: {best_version_mse})")
        except Exception:
            best_version_mse = float('inf')
            self.logger.warning(f"⚠️ No existing 'best' model version found.")

        # ✅ Compare MSE and update alias if the new model is better
        self.logger.info(f"🔍 Comparing current MSE ({self.mse}) with 'best' version MSE ({best_version_mse})")
        if self.mse < best_version_mse:
            client.set_model_version_tag(model_version.name, model_version.version, "mse", str(self.mse))
            client.set_registered_model_alias(self.model_name, "best", model_version.version)
            self.logger.info(f"🏆 'best' alias updated to version {model_version.version} with MSE {self.mse}")
        else:
            self.logger.info(f"📉 Model version {model_version.version} not assigned 'best' alias (MSE is worse).")

    def execute_training(self, state: WorkflowState) -> WorkflowState:
        """
        Executes full training pipeline and updates WorkflowState.
        """
        df = self.fetch_data_from_db(state.database_url)
        X, y = self.preprocess_data(df, self.get_granularity(state.date_period))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_model_with_hyperparameter_tuning(X_train, y_train)
        self.evaluate_and_log_model(X_test, y_test)

        return state.model_copy(update={"mse": self.mse})
