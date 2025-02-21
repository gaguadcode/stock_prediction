import pandas as pd
import numpy as np
import pickle
import redis
from typing import List
from app.utils.logger import get_logger
from app.utils.datatypes import FinalPredictionState, StockPredictionRequest, StockPredictionOutput
from sklearn.ensemble import GradientBoostingRegressor


class StockPredictor:
    """
    Uses a trained Gradient Boosting model from FinalPredictionState 
    to make predictions on future stock prices.
    """

    def __init__(self, final_state: FinalPredictionState, redis_host="localhost", redis_port=6379, redis_db=0):
        """
        Initialize the predictor with a trained model and relevant metadata.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.final_state = final_state
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

        # ✅ Extract granularity from final_state (date_period)
        self.granularity = final_state.date_period

        # Load trained model from Redis
        self.model = self.load_model_from_redis()

        if not isinstance(self.model, GradientBoostingRegressor):
            raise ValueError("The model inside FinalPredictionState must be a trained GradientBoostingRegressor!")

    def load_model_from_redis(self, model_key="trained_model") -> GradientBoostingRegressor:
        """
        Loads the trained Gradient Boosting model from Redis.
        """
        model_bytes = self.redis_client.get(model_key)
        if model_bytes is None:
            self.logger.error("No trained model found in Redis.")
            raise ValueError("No trained model found in Redis. Ensure that the model has been trained and saved.")

        self.logger.info("Loading trained model from Redis...")
        return pickle.loads(model_bytes)

    def transform_dates(self, stock_prediction_request: StockPredictionRequest) -> pd.DataFrame:
        """
        Converts the target dates from StockPredictionRequest into numerical features,
        applying the correct granularity (monthly, weekly, or daily).
        """
        try:
            self.logger.info("Transforming target dates into numerical features...")
            dates = pd.to_datetime(stock_prediction_request.date_target)
            df = pd.DataFrame({'date': dates})

            df['year'] = df['date'].dt.year  # Always include year for temporal context

            # ✅ Preserve `month`, `week`, or `day` for model consistency
            if self.granularity == "TIME_SERIES_MONTHLY":
                df['month'] = df['date'].dt.month  # 🔹 Keep raw month column
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            elif self.granularity == "TIME_SERIES_WEEKLY":
                df['week'] = df['date'].dt.isocalendar().week  # 🔹 Keep raw week column
                df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
                df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

            elif self.granularity == "TIME_SERIES_DAILY":
                df['day'] = df['date'].dt.day  # 🔹 Keep raw day column
                df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
                df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

            # ✅ Drop only the original date column
            df = df.drop(columns=['date'])

            self.logger.info("Date transformation complete.")
            return df
        except Exception as e:
            self.logger.error(f"Error in transform_dates: {e}")
            raise


    def make_predictions(self, stock_prediction_request: FinalPredictionState) -> StockPredictionOutput:
        """
        Predicts stock prices based on the provided StockPredictionRequest.

        Args:
            stock_prediction_request (StockPredictionRequest): The input data with stock symbol, 
            date period, and target dates.

        Returns:
            StockPredictionOutput: A structured output with metadata and predictions.
        """
    

        # ✅ Transform target dates using the correct granularity
        transformed_data = self.transform_dates(stock_prediction_request)

        # Use trained model to make predictions
        self.logger.info(f"Making predictions for {stock_prediction_request.stock_symbol}...")
        predictions = self.model.predict(transformed_data).tolist()
        self.logger.info("Stock Prediction Output: %s", {
            "user_input": stock_prediction_request.user_input,  # Keep user query
            "stock_symbol": stock_prediction_request.stock_symbol,
            "date_period": stock_prediction_request.date_period,
            "date_target": stock_prediction_request.date_target,
            "database_url": stock_prediction_request.database_url,  # Retain DB metadata
            "mse": stock_prediction_request.mse,  # Retain model evaluation score
            "predictions": predictions
        })
        # Return structured output
        return StockPredictionOutput(
            user_input=stock_prediction_request.user_input,  # Keep user query
            stock_symbol=stock_prediction_request.stock_symbol,
            date_period=stock_prediction_request.date_period,
            date_target=stock_prediction_request.date_target,
            database_url=stock_prediction_request.database_url,  # Retain DB metadata
            mse=stock_prediction_request.mse,  # Retain model evaluation score
            predictions=predictions
        )
