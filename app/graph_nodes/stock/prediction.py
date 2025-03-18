import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from typing import List
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState
from sklearn.ensemble import GradientBoostingRegressor
from app.utils.utils import anonymize_database_url


class StockPredictor:
    """
    Uses the latest "best" trained Gradient Boosting model from MLflow 
    to make predictions on future stock prices.
    """

    def __init__(self, state: WorkflowState):
        """
        Initializes the predictor with a trained model from MLflow.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.state = state
        self.mlflow_client = MlflowClient()  # ✅ Initialize MLflow client
        self.model_alias = "best"  # ✅ Load by alias only
        self.model_name = "GradientBoostingRegressor"  # ✅ Ensure consistency

        # ✅ Extract granularity from state
        self.granularity = state.date_period

        # ✅ Load trained model from MLflow (Ensures model must exist)
        self.model = self.load_model_from_mlflow()

        if not isinstance(self.model, GradientBoostingRegressor):
            raise ValueError("❌ The loaded model from MLflow is not a GradientBoostingRegressor!")

    def load_model_from_mlflow(self) -> GradientBoostingRegressor:
        """
        Loads the latest trained Gradient Boosting model from MLflow Model Registry 
        using the alias 'best'. Raises an exception if no model is found.
        """
        try:
            self.logger.info("📥 Attempting to load the latest trained model from MLflow (alias: 'best')...")

            # ✅ Load the model using alias
            model_uri = f"models:/{self.model_name}@{self.model_alias}"
            model = mlflow.sklearn.load_model(model_uri)

            self.logger.info(f"✅ Successfully loaded latest trained model from MLflow using alias '{self.model_alias}'.")
            return model

        except mlflow.exceptions.MlflowException as e:
            self.logger.error(f"❌ MLflow error: {e}")
            raise ValueError("❌ MLflow Model Registry is unavailable or misconfigured.")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model from MLflow: {e}")
            raise ValueError("❌ No trained model found in MLflow with alias 'best'. Ensure training has been completed.")

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
        Raises an exception if the model is not found.
        """
        if not self.state.date_target:
            raise ValueError("❌ Target dates are missing. Ensure entity extraction is complete before prediction.")

        # ✅ Transform target dates
        transformed_data = self.transform_dates()

        # ✅ Use trained model to make predictions
        self.logger.info(f"📊 Making predictions for {self.state.stock_symbol}...")
        predictions = list(map(float, self.model.predict(transformed_data)))  # Ensure floats

        # ✅ Dump full state, but anonymize database_url before logging
        state_dict = self.state.model_dump()
        if "database_url" in state_dict:
            state_dict["database_url"] = anonymize_database_url(state_dict["database_url"])

        # ✅ Add predictions to the state before logging
        state_dict["predictions"] = predictions

        self.logger.info("✅ Stock Prediction Output: %s", state_dict)

        # ✅ Avoid duplicate key error by updating only if necessary
        updated_data = {}
        if self.state.predictions is None:
            updated_data["predictions"] = predictions  # Add predictions only if missing

        # ✅ Convert to WorkflowState explicitly
        new_state = self.state.model_copy(update=updated_data)

        # ✅ Ensure the return type is correct
        if not isinstance(new_state, WorkflowState):
            raise TypeError(f"❌ make_predictions() returned {type(new_state)} instead of WorkflowState")

        return new_state