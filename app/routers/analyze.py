from fastapi import APIRouter, HTTPException
from app.utils.datatypes import AnalysisRequest, AnalysisResponse
from app.graph_nodes import stock_extractor_agent

router = APIRouter()

@router.post("/", response_model=AnalysisResponse)
async def analyze_text(data: AnalysisRequest):
    try:
        analysis_summary = stock_extractor_agent.analyze_text(data.text)
        return AnalysisResponse(analysis_summary=analysis_summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
