import torch
from app.config import config

class StockModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict(self, stock_data):
        with torch.no_grad():
            prediction = self.model(stock_data)
        return prediction.item()

stock_model = StockModel(config.MODEL_PATH)

def predict(stock_data):
    return stock_model.predict(stock_data)
