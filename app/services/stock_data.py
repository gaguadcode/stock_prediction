import requests
import pandas as pd
import os
from app.config import config

# Load environment variables from .env file


# Get the base URL and API key from the config
STOCK_URL = config.ALPHAVANTAGE_BASE_URL
API_KEY = config.ALPHAVANTAGE_API_KEY
OUTPUT_DATA = config.OUTPUT_CSV

async def fetch_and_save_historical_data(stock_symbol: str, interval: str):
    """
    Fetch historical data from the API, process it, and save it to a CSV file.

    Args:
        api_url (str): The URL for fetching historical data.
        output_csv (str): The file path to save the processed data as a CSV.
    
    Returns:
        pd.DataFrame: The processed data as a Pandas DataFrame.
    """
    try:
        url = f"{STOCK_URL}?function={stock_symbol}&interval={interval}&apikey={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        structured_data = [{'month': entry['date'], 'price': entry['value']} for entry in data['data']]

        df = pd.DataFrame(structured_data)

        df.to_csv(OUTPUT_DATA, index=False)
        print(f"Data saved to {OUTPUT_DATA}")

        return df
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        raise
    except KeyError as e:
        print(f"Error processing data: {e}")
        raise

