from pydantic import BaseModel, Field
from typing import Literal

class StockPredictionRequest(BaseModel):
    stock_symbol: str
    date_period: Literal["monthly", "weekly", "daily"]  # Only these values are valid
    date_target: str = Field(
        ...,  # Required field
        regex=r"^\d{4}-\d{2}-\d{2}$",  # Matches the "YYYY-MM-DD" format
        description="Target date in 'YYYY-MM-DD' format (e.g., '2025-01-01').",
    )

class AnalysisResponse(BaseModel):
    generative_response: str
    stock_symbol: str
    date_period: Literal["monthly", "weekly", "daily"]  # Only these values are valid
    date_target: str = Field(
        ...,  # Required field
        regex=r"^\d{4}-\d{2}-\d{2}$",  # Matches the "YYYY-MM-DD" format
        description="Target date in 'YYYY-MM-DD' format (e.g., '2025-01-01').",
    )
