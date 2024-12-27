import requests
import pandas as pd
from app.utils.config import config
from app.utils.datatypes import StockPredictionRequest
from app.utils.logger import get_logger


class HistoricalDataFetcher:
    """
    A class to fetch historical stock data from an external API, validate inputs, process the data,
    and save it to a CSV file.
    """

    def __init__(self):
        """
        Initialize the HistoricalDataFetcher with configuration settings.
        """
        self.logger = get_logger(self.__class__.__name__)  # Initialize a logger for this class
        self.base_url = config.ALPHAVANTAGE_BASE_URL
        self.api_key = config.ALPHAVANTAGE_API_KEY
        self.output_data = config.OUTPUT_CSV
        self.logger.info("HistoricalDataFetcher initialized.")

    def validate_input(self, agent_output: dict) -> StockPredictionRequest:
        """
        Validate the input dictionary using Pydantic.

        Args:
            agent_output (dict): Input dictionary containing stock details.

        Returns:
            StockPredictionRequest: Validated Pydantic object.
        """
        try:
            self.logger.info("Validating input...")
            validated_request = StockPredictionRequest(**agent_output)
            self.logger.info("Input validation successful.")
            return validated_request
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            raise ValueError(f"Input validation error: {e}")

    def construct_api_url(self, stock_symbol: str, interval: str) -> str:
        """
        Construct the API URL using the stock symbol and interval.

        Args:
            stock_symbol (str): Stock symbol (e.g., 'AAPL').
            interval (str): Time interval ('daily', 'weekly', 'monthly').

        Returns:
            str: The constructed API URL.
        """
        url = f"{self.base_url}?function={interval}&symbol={stock_symbol.upper()}&apikey={self.api_key}"
        self.logger.debug(f"Constructed API URL: {url}")
        return url

    def fetch_data_from_api(self, url: str) -> dict:
        """
        Fetch data from the external API.

        Args:
            url (str): API endpoint URL.

        Returns:
            dict: JSON response from the API.

        Raises:
            requests.RequestException: If the API call fails.
        """
        try:
            self.logger.info(f"Fetching data from API: {url}")
            response = requests.get(url)
            response.raise_for_status()
            self.logger.info("Data fetched successfully from API.")
            self.logger.debug(f"API Response: {response.json()}")
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data from API: {e}")
            raise

    def process_api_response(self, api_response: dict) -> pd.DataFrame:
        """
        Process and structure the API response into a Pandas DataFrame.

        Args:
            api_response (dict): JSON response from the API.

        Returns:
            pd.DataFrame: Processed data as a DataFrame.
        """
        try:
            self.logger.info("Processing API response...")
            time_series = api_response.get("Monthly Time Series", {})
            data = [
                {"date": date, "price": float(info["1. open"])}
                for date, info in time_series.items()
            ]
            df = pd.DataFrame(data)
            self.logger.info("API response processed successfully.")
            self.logger.debug(f"Processed DataFrame:\n{df.head()}")
            return df
        except KeyError as e:
            self.logger.error(f"Error processing data: Missing key {e}")
            raise KeyError(f"Error processing data: Missing key {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error while processing data: {e}")
            raise ValueError(f"Unexpected error while processing data: {e}")

    def save_to_csv(self, df: pd.DataFrame):
        """
        Save the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
        """
        try:
            df.to_csv(self.output_data, index=False)
            self.logger.info(f"Data saved to {self.output_data}")
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}")
            raise

    async def fetch_and_save_historical_data(self, agent_output: dict) -> pd.DataFrame:
        """
        Complete workflow: Validate input, fetch data, process it, and save it to a CSV file.

        Args:
            agent_output (dict): Input dictionary containing stock details.

        Returns:
            pd.DataFrame: The processed data as a Pandas DataFrame.
        """
        try:
            self.logger.info("Starting data fetching workflow...")

            # Step 1: Validate input
            validated_request = self.validate_input(agent_output)

            # Step 2: Construct API URL
            url = self.construct_api_url(
                stock_symbol=validated_request.stock_symbol,
                interval=validated_request.date_period,
            )

            # Step 3: Fetch data from API
            self.logger.info("Fetching historical stock data...")
            api_response = self.fetch_data_from_api(url)

            # Step 4: Process API response
            self.logger.info("Processing the fetched data...")
            df = self.process_api_response(api_response)

            # Step 5: Save to CSV
            self.logger.info("Saving processed data to CSV...")
            self.save_to_csv(df)

            self.logger.info("Data fetching workflow completed successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error in data fetching workflow: {e}")
            raise
