import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from typing import List
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState
from sklearn.ensemble import GradientBoostingRegressor


class StockPredictor:
    """
    Uses a trained Gradient Boosting model from WorkflowState 
    to make predictions on future stock prices.
    """

    def __init__(self, state: WorkflowState):
        """
        Initializes the predictor with a trained model from MLflow.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.state = state
        self.mlflow_client = MlflowClient()  # ✅ Initialize MLflow client
        self.model_name = "GradientBoostingRegressor"  # ✅ Ensure consistency

        # ✅ Extract granularity from state
        self.granularity = state.date_period

        # ✅ Load trained model from MLflow artifacts
        self.model = self.load_model_from_mlflow()

        if not isinstance(self.model, GradientBoostingRegressor):
            raise ValueError("The loaded model from MLflow is not a GradientBoostingRegressor!")

    def load_model_from_mlflow(self) -> GradientBoostingRegressor:
        """
        Loads the latest trained Gradient Boosting model from MLflow Model Registry.
        """
        try:
            self.logger.info("📥 Attempting to load the latest trained model from MLflow...")

            # ✅ Ensure MLflow tracking server is reachable
            registered_models = self.mlflow_client.search_registered_models()
            model_names = [model.name for model in registered_models]

            if self.model_name not in model_names:
                self.logger.error(f"❌ Model '{self.model_name}' not found in MLflow Registry.")
                raise ValueError("No registered model found in MLflow. Ensure training has been completed.")

            # ✅ Fetch latest model version
            model_version = self.mlflow_client.get_model_version_by_alias(self.model_name, "latest")

            if not model_version:
                self.logger.error("❌ No model version found under alias 'latest'.")
                raise ValueError("No model version assigned to 'latest'. Ensure training has been completed.")

            # ✅ Load the model using the correct URI
            model_uri = f"models:/{self.model_name}@latest"
            model = mlflow.sklearn.load_model(model_uri)

            self.logger.info(f"✅ Successfully loaded latest trained model from MLflow (version: {model_version.version}).")
            return model

        except mlflow.exceptions.RestException as e:
            self.logger.error(f"❌ MLflow error: {e}")
            raise ValueError("MLflow Model Registry is unavailable or misconfigured.")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model from MLflow: {e}")
            raise ValueError("No trained model found in MLflow. Ensure that the model has been trained and logged.")

    def transform_dates(self) -> pd.DataFrame:
        """
        Converts the target dates into numerical features, applying the correct granularity.
        """
        try:
            self.logger.info("🔄 Transforming target dates into numerical features...")
            dates = pd.to_datetime(self.state.date_target)
            df = pd.DataFrame({'date': dates})

            df['year'] = df['date'].dt.year  # Always include year for temporal context

            # ✅ Preserve `month`, `week`, or `day` for model consistency
            if self.granularity == "TIME_SERIES_MONTHLY":
                df['month'] = df['date'].dt.month
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            elif self.granularity == "TIME_SERIES_WEEKLY":
                df['week'] = df['date'].dt.isocalendar().week
                df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
                df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

            elif self.granularity == "TIME_SERIES_DAILY":
                df['day'] = df['date'].dt.day
                df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
                df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

            # ✅ Drop only the original date column
            df = df.drop(columns=['date'])

            self.logger.info("✅ Date transformation complete.")
            return df
        except Exception as e:
            self.logger.error(f"⚠️ Error in transform_dates: {e}")
            raise

    def make_predictions(self) -> WorkflowState:
        """
        Predicts stock prices based on the transformed state.
        Ensures that `predictions` is only updated if not already set.
        """
        if not self.state.date_target:
            raise ValueError("❌ Target dates are missing. Ensure entity extraction is complete before prediction.")

        # ✅ Transform target dates
        transformed_data = self.transform_dates()

        # ✅ Use trained model to make predictions
        self.logger.info(f"📊 Making predictions for {self.state.stock_symbol}...")
        predictions = list(map(float, self.model.predict(transformed_data)))  # Ensure floats

        self.logger.info("✅ Stock Prediction Output: %s", {
            "user_input": self.state.user_input,  
            "stock_symbol": self.state.stock_symbol,
            "date_period": self.state.date_period,
            "date_target": self.state.date_target,
            "database_url": self.state.database_url,  
            "mse": self.state.mse,  
            "predictions": predictions
        })

        # ✅ Avoid duplicate key error by updating only if necessary
        updated_data = {}
        if self.state.predictions is None:
            updated_data["predictions"] = predictions  # Add predictions only if missing

        # ✅ Convert to WorkflowState explicitly
        new_state = self.state.model_copy(update=updated_data)

        # ✅ Check type
        if not isinstance(new_state, WorkflowState):
            raise TypeError(f"❌ make_predictions() returned {type(new_state)} instead of WorkflowState")

        return new_state
