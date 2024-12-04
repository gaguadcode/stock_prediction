from fastapi import FastAPI
from app.routers import predict, analyze

app = FastAPI()

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])
