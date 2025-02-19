from pydantic_settings import BaseSettings
class Config(BaseSettings):
    ALPHAVANTAGE_API_KEY: str
    ALPHAVANTAGE_BASE_URL: str
    OUTPUT_CSV: str
    langsmith_tracing: bool  # Change to correct type (bool instead of str if applicable)
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str

    class Config:
        env_file = ".env"

config = Config()
