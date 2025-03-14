from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional
from datetime import datetime

# ✅ Shared validation function for date_target
def validate_date_target(value):
    if not isinstance(value, list):
        raise ValueError("date_target must be a list of dates in 'YYYY-MM-DD' format.")
    for date_str in value:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD'.")
    return value

class WorkflowState(BaseModel):
    """
    Unified state model to track all transformations in the workflow.
    This ensures LangGraph properly updates state across nodes.
    """

    # ✅ Step 1: User Input
    next_state: Optional[str] = None
    user_input: Optional[str] = None

    # ✅ Step 2: Stock Prediction Pipeline
    stock_symbol: Optional[str] = None
    date_period: Optional[Literal["TIME_SERIES_MONTHLY", "TIME_SERIES_WEEKLY", "TIME_SERIES_DAILY"]] = None
    date_target: Optional[List[str]] = Field(
        None, description="A list of target dates in 'YYYY-MM-DD' format."
    )

    @field_validator("date_target", mode="before")
    def validate_date_target(cls, value):
        if value is not None:
            return validate_date_target(value)
        return value

    database_url: Optional[str] = None
    mse: Optional[float] = None  # Mean Squared Error of the trained model

    # ✅ Final Step: Predictions
    predictions: Optional[List[float]] = None
    research_output: Optional[str] = None
    reasoning_output: Optional[str] = None
