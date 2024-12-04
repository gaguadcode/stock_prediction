from pydantic import BaseSettings

class Config(BaseSettings):
    STOCK_API_KEY: str
    STOCK_URL: str
    DATABASE_URL: str

    class Config:
        env_file = ".env"

config = Config()
