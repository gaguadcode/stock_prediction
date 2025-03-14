import pandas as pd
import numpy as np
import pickle
import redis
from typing import List
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState
from sklearn.ensemble import GradientBoostingRegressor


class StockPredictor:
    """
    Uses a trained Gradient Boosting model from WorkflowState 
    to make predictions on future stock prices.
    """

    def __init__(self, state: WorkflowState, redis_host="localhost", redis_port=6379, redis_db=0):
        """
        Initializes the predictor with a trained model and relevant metadata.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.state = state
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

        # ✅ Extract granularity from state
        self.granularity = state.date_period

        # ✅ Load trained model from Redis
        self.model = self.load_model_from_redis()

        if not isinstance(self.model, GradientBoostingRegressor):
            raise ValueError("The model inside WorkflowState must be a trained GradientBoostingRegressor!")

    def load_model_from_redis(self, model_key="trained_model") -> GradientBoostingRegressor:
        """
        Loads the trained Gradient Boosting model from Redis.
        """
        model_bytes = self.redis_client.get(model_key)
        if model_bytes is None:
            self.logger.error("No trained model found in Redis.")
            raise ValueError("No trained model found in Redis. Ensure that the model has been trained and saved.")

        self.logger.info("✅ Loading trained model from Redis...")
        return pickle.loads(model_bytes)

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