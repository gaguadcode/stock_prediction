import requests
import pandas as pd
import asyncio
import asyncpg
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Importing models
from app.utils.config import config
from app.utils.datatypes import EntityExtractOutput, DataFetchOutput
from app.utils.logger import get_logger


class HistoricalDataFetcher:
    """
    A class to fetch, process, and store historical stock data based on EntityExtractOutput.
    """

    def __init__(self):
        """
        Initialize the HistoricalDataFetcher with configuration settings.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.base_url = config.ALPHAVANTAGE_BASE_URL
        self.api_key = config.ALPHAVANTAGE_API_KEY
        self.postgres_url = "postgresql://gustavo:password@localhost/postgres"
        self.logger.info("HistoricalDataFetcher initialized.")

    async def create_database_if_not_exists(self, db_name: str):
        """
        Asynchronously create the database if it does not already exist.
        """
        try:
            self.logger.info(f"Checking if database '{db_name}' exists...")

            # Connect to the 'postgres' database
            conn = await asyncpg.connect(dsn="postgresql://gustavo:password@localhost/postgres")

            # Check if the database exists
            exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1;", db_name)

            if not exists:
                self.logger.info(f"Creating database: {db_name}")
                await conn.execute(f'CREATE DATABASE "{db_name}";')
                self.logger.info(f"Database '{db_name}' created successfully.")

            # Close the connection
            await conn.close()

        except Exception as e:
            self.logger.error(f"Error creating database '{db_name}': {e}")
            raise

    def construct_api_url(self, stock_symbol: str, interval: str) -> str:
        """
        Construct the API URL using the stock symbol and interval.
        """
        url = f"{self.base_url}?function={interval}&symbol={stock_symbol.upper()}&apikey={self.api_key}"
        self.logger.debug(f"Constructed API URL: {url}")
        return url

    def fetch_data_from_api(self, url: str) -> dict:
        """
        Fetch data from the external API.
        """
        try:
            self.logger.info(f"Fetching data from API: {url}")
            response = requests.get(url)
            response.raise_for_status()
            self.logger.info("Data fetched successfully from API.")
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data from API: {e}")
            raise

    def process_api_response(self, api_response: dict, stock_symbol: str) -> pd.DataFrame:
        """
        Process and structure the API response into a Pandas DataFrame.
        """
        try:
            self.logger.info("Processing API response...")
            time_series = api_response.get("Monthly Time Series", {})

            if not time_series:
                raise KeyError("Missing 'Monthly Time Series' in API response.")

            data = [
                {"date": datetime.strptime(date, "%Y-%m-%d"), "price": float(info["1. open"]), "stock_symbol": stock_symbol}
                for date, info in time_series.items()
            ]
            df = pd.DataFrame(data)
            self.logger.info("API response processed successfully.")
            return df
        except KeyError as e:
            self.logger.error(f"Error processing data: Missing key {e}")
            raise KeyError(f"Error processing data: Missing key {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error while processing data: {e}")
            raise ValueError(f"Unexpected error while processing data: {e}")

    def store_to_database(self, df: pd.DataFrame, db_name: str):
        """
        Stores the processed DataFrame into a dynamically created PostgreSQL database.
        """
        db_url = f"postgresql://gustavo:password@localhost/{db_name}"
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        Base = declarative_base()

        class StockHistoricalData(Base):
            __tablename__ = "historical_stock_data"

            id = Column(String, primary_key=True)
            stock_symbol = Column(String, nullable=False)
            date = Column(DateTime, nullable=False)
            price = Column(Float, nullable=False)
            date_fetched = Column(DateTime, default=datetime.utcnow)

        Base.metadata.create_all(engine)

        db_session = SessionLocal()
        try:
            self.logger.info(f"Storing data in database {db_name}...")

            for _, row in df.iterrows():
                record = StockHistoricalData(
                    id=str(row["date"].strftime("%Y%m%d") + row["stock_symbol"]),
                    stock_symbol=row["stock_symbol"],
                    date=row["date"],
                    price=row["price"]
                )

                # Use `merge()` to update existing records instead of causing duplicates
                db_session.merge(record)

            db_session.commit()
            self.logger.info(f"Data successfully stored in database {db_name}.")

        except Exception as e:
            db_session.rollback()
            self.logger.error(f"Database error in {db_name}: {e}")

        finally:
            db_session.close()

    def fetch_and_store_historical_data(self, entity_extract_output: EntityExtractOutput) -> DataFetchOutput:
        """
        Process EntityExtractOutput, fetch data, store in DB, and return DataFetchOutput.
        """
        self.logger.info("Processing entity extract output...")

        # Construct database name based on stock symbol and time series type
        db_name = f"{entity_extract_output.stock_symbol.lower()}_{entity_extract_output.date_period.lower()}"

        # Run async database creation synchronously
        asyncio.run(self.create_database_if_not_exists(db_name))

        # Fetch data
        url = self.construct_api_url(entity_extract_output.stock_symbol, entity_extract_output.date_period)
        api_response = self.fetch_data_from_api(url)
        df = self.process_api_response(api_response, entity_extract_output.stock_symbol)

        # Store to database
        self.store_to_database(df, db_name)

        # Construct database URL
        database_url = f"postgresql://gustavo:password@localhost/{db_name}"

        # Return DataFetchOutput
        return DataFetchOutput(
        user_input=entity_extract_output.user_input,  # Keep user input
        stock_symbol=entity_extract_output.stock_symbol,
        date_period=entity_extract_output.date_period,
        date_target=entity_extract_output.date_target,
        database_url=database_url)
