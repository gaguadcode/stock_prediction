from pydantic_settings import BaseSettings
class Config(BaseSettings):
    ALPHAVANTAGE_API_KEY: str
    ALPHAVANTAGE_BASE_URL: str
    OUTPUT_CSV: str

    class Config:
        env_file = ".env"

config = Config()
