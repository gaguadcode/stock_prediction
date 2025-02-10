import pandas as pd
import numpy as np
from typing import List
from app.utils.logger import get_logger
from app.utils.datatypes import FinalPredictionState, StockPredictionRequest, StockPredictionOutput
from sklearn.ensemble import GradientBoostingRegressor


class StockPredictor:
    """
    Uses a trained Gradient Boosting model from FinalPredictionState 
    to make predictions on future stock prices.
    """

    def __init__(self, final_state: FinalPredictionState):
        """
        Initialize the predictor with a trained model and relevant metadata.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.final_state = final_state
        self.model = final_state.model

        if not isinstance(self.model, GradientBoostingRegressor):
            raise ValueError("The model inside FinalPredictionState must be a trained GradientBoostingRegressor!")

    def transform_dates(self, stock_prediction_request: StockPredictionRequest) -> pd.DataFrame:
        """
        Converts the target dates from StockPredictionRequest into numerical features.
        """
        try:
            self.logger.info("Transforming date column into features...")
            dates = pd.to_datetime(stock_prediction_request.date_target)
            df = pd.DataFrame({'date': dates})

            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            df = df.drop(columns=['date'])
            return df
        except Exception as e:
            self.logger.error(f"Error in transform_dates: {e}")
            raise

    def make_predictions(self, stock_prediction_request: StockPredictionRequest) -> StockPredictionOutput:
        """
        Predicts stock prices based on the provided StockPredictionRequest.

        Args:
            stock_prediction_request (StockPredictionRequest): The input data with stock symbol, 
            date period, and target dates.

        Returns:
            StockPredictionOutput: A structured output with metadata and predictions.
        """
        if not isinstance(stock_prediction_request, StockPredictionRequest):
            raise ValueError("Input must be a StockPredictionRequest object!")

        # Transform dates into ML-friendly features
        transformed_data = self.transform_dates(stock_prediction_request)

        # Use trained model to make predictions
        self.logger.info(f"Making predictions for {stock_prediction_request.stock_symbol}...")
        predictions = self.model.predict(transformed_data).tolist()

        # Return structured output
        return StockPredictionOutput(
            entity_extract=self.final_state,  # Retain full metadata
            model=self.model,  # Keep model in the output
            predictions=predictions
        )
