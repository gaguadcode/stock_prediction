import requests
import pandas as pd
from app.config import config
from utils.datatypes import StockPredictionRequest


class HistoricalDataFetcher:
    """
    A class to fetch historical stock data from an external API, validate inputs, process the data,
    and save it to a CSV file.
    """

    def __init__(self):
        """
        Initialize the HistoricalDataFetcher with configuration settings.
        """
        self.base_url = config.ALPHAVANTAGE_BASE_URL
        self.api_key = config.ALPHAVANTAGE_API_KEY
        self.output_data = config.OUTPUT_CSV

    def validate_input(self, agent_output: dict) -> StockPredictionRequest:
        """
        Validate the input dictionary using Pydantic.

        Args:
            agent_output (dict): Input dictionary containing stock details.

        Returns:
            StockPredictionRequest: Validated Pydantic object.
        """
        try:
            return StockPredictionRequest(**agent_output)
        except Exception as e:
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
        return f"{self.base_url}?function=TIME_SERIES_{interval.upper()}&symbol={stock_symbol}&apikey={self.api_key}"

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
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching data from API: {e}")

    def process_api_response(self, api_response: dict) -> pd.DataFrame:
        """
        Process and structure the API response into a Pandas DataFrame.

        Args:
            api_response (dict): JSON response from the API.

        Returns:
            pd.DataFrame: Processed data as a DataFrame.

        Raises:
            KeyError: If the expected keys are missing in the response.
        """
        try:
            # Adjust the parsing logic according to the API's response structure
            data = [
                {'date': entry['date'], 'price': entry['value']}
                for entry in api_response['data']
            ]
            return pd.DataFrame(data)
        except KeyError as e:
            raise KeyError(f"Error processing data: Missing key {e}")

    def save_to_csv(self, df: pd.DataFrame):
        """
        Save the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
        """
        df.to_csv(self.output_data, index=False)
        print(f"Data saved to {self.output_data}")

    async def fetch_and_save_historical_data(self, agent_output: dict) -> pd.DataFrame:
        """
        Complete workflow: Validate input, fetch data, process it, and save it to a CSV file.

        Args:
            agent_output (dict): Input dictionary containing stock details.

        Returns:
            pd.DataFrame: The processed data as a Pandas DataFrame.
        """
        try:
            # Step 1: Validate input
            validated_request = self.validate_input(agent_output)

            # Step 2: Construct API URL
            url = self.construct_api_url(
                stock_symbol=validated_request.stock_symbol,
                interval=validated_request.date_period,
            )

            # Step 3: Fetch data from API
            print(f"Fetching data from: {url}")
            api_response = self.fetch_data_from_api(url)

            # Step 4: Process API response
            df = self.process_api_response(api_response)

            # Step 5: Save to CSV
            self.save_to_csv(df)

            return df

        except Exception as e:
            print(f"Error in data fetching workflow: {e}")
            raise



