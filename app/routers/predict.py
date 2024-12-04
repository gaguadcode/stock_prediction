from fastapi import APIRouter, HTTPException
from app.utils.datatypes import StockPredictionRequest, StockPredictionResponse
from app.graph_nodes import stock_data, model_inference

router = APIRouter()

@router.post("/", response_model=StockPredictionResponse)
async def predict_stock(data: StockPredictionRequest):
    try:
        stock_info = await stock_data.fetch_stock_data(data.stock_symbol, data.date)
        predicted_price = model_inference.predict(stock_info)
        return StockPredictionResponse(
            stock_symbol=data.stock_symbol, date=data.date, predicted_price=predicted_price
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
