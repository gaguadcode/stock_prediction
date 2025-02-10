from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.graph_nodes.stock import stock_extractor_agent  # Assuming it extracts data from text
from utils.datatypes import AnalysisResponse
router = APIRouter()


@router.post("/", response_model=AnalysisResponse)
async def analyze_text(data: str):
    """
    Accepts a plain string input and returns a validated response conforming to AnalysisResponse.
    """
    try:
        # Step 1: Process the text with the agent
        analysis_result = stock_extractor_agent.analyze_text(data)
        
        # Step 2: Validate and return the response
        # Assuming the `stock_extractor_agent.analyze_text` returns a dictionary
        validated_response = AnalysisResponse(**analysis_result)
        return validated_response

    except ValueError as ve:
        # Raised if the response doesn't conform to AnalysisResponse
        raise HTTPException(status_code=400, detail=f"Validation Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
