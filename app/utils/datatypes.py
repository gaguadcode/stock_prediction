from pydantic import BaseModel, Field, field_validator
from typing import Literal, List
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from typing import Any

# Shared validation function for date_target
def validate_date_target(value):
    """
    Validates that the input is a list of dates in 'YYYY-MM-DD' format.

    Args:
        value (List[str]): The list of dates to validate.

    Returns:
        List[str]: The validated list of dates.

    Raises:
        ValueError: If the value is not a list or if any date in the list is invalid.
    """
    if not isinstance(value, list):
        raise ValueError("date_target must be a list of dates in 'YYYY-MM-DD' format.")
    for date_str in value:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD'.")
    return value

class UserInputString(BaseModel):
    user_input: str

class ResearchOutput(BaseModel):
    user_input: UserInputString
    research_output: str
# StockPredictionRequest model
class ReasoningOutput(BaseModel):
    user_input: UserInputString
    reasoning_output: str
    
class StockPredictionRequest(BaseModel): 
    stock_symbol: str
    date_period: Literal["TIME_SERIES_MONTHLY", "TIME_SERIES_WEEKLY", "TIME_SERIES_DAILY"]
    date_target: List[str] = Field(
        ...,  
        description="A list of target dates in 'YYYY-MM-DD' format (e.g., ['2025-01-01', '2025-02-01']).",
    )

    @field_validator("date_target", mode="before")
    def validate_date_target(cls, value):
        return validate_date_target(value)

class EntityExtractOutput(BaseModel):
    user_input: str
    stock_prediction: StockPredictionRequest

class DataFetchOutput(BaseModel):
    entity_extract:EntityExtractOutput
    database_url: str

class FinalPredictionState(BaseModel):
    entity_extract: DataFetchOutput
    mse: float  # Mean Squared Error of the model

class StockPredictionOutput(BaseModel):
    """
    This model holds the full prediction output, including metadata, model, 
    and predictions.
    """
    entity_extract: FinalPredictionState  # Contains all necessary metadata
    predictions: List[float]  # The predicted stock prices

'''
# AnalysisResponse model
class AnalysisResponse(BaseModel):
    generative_response: str
    stock_symbol: str
    date_period: Literal["TIME_SERIES_MONTHLY", "TIME_SERIES_WEEKLY", "TIME_SERIES_DAILY"]  # Only these values are valid
    date_target: List[str] = Field(
        ...,  # Required field
        description="A list of target dates in 'YYYY-MM-DD' format (e.g., ['2025-01-01', '2025-02-01']).",
    )

    @field_validator("date_target", mode="before")
    def validate_date_target(cls, value):
        return validate_date_target(value)
'''