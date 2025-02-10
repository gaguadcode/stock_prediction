from fastapi import APIRouter, HTTPException
from app.graph_nodes.stock import stock_fetch_data
from app.utils.datatypes import StockPredictionRequest, StockPredictionResponse
from app.graph_nodes.stock import model_training

router = APIRouter()

@router.post("/", response_model=StockPredictionResponse)
async def predict_stock(data: StockPredictionRequest):
    try:
        stock_info = await stock_fetch_data.fetch_stock_data(data.stock_symbol, data.date)
        predicted_price = model_training.predict(stock_info)
        return StockPredictionResponse(
            stock_symbol=data.stock_symbol, date=data.date, predicted_price=predicted_price
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
