import json
import requests
import pandas as pd
from app.config import config
from app.models import AnalysisResponse  # Assuming your response model is defined here
from app.graph_nodes.prediction_model import PredictionModel  # Import your prediction model
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


def initialize_langchain_agent():
    """
    Initialize the LangChain agent.
    """
    llm = OpenAI(temperature=0)  # Use your preferred LLM
    return llm


def process_natural_language(input_text: str):
    """
    Process the input natural language to extract stock symbol, date period, and target date.

    Args:
        input_text (str): The natural language input from the user.

    Returns:
        dict: Extracted data with stock_symbol, date_period, and date_target.
    """
    agent = initialize_langchain_agent()
    prompt_template = PromptTemplate(
        input_variables=["input_text"],
        template=(
            "Extract the stock symbol (company or commodity), prediction window "
            "(monthly, weekly, daily...), and target date of prediction from the following input:\n"
            "{input_text}\n"
            "Respond with JSON format: {{'stock_symbol': '...', 'date_period': '...', 'date_target':'...'}}"
        ),
    )
    prompt = prompt_template.format(input_text=input_text)
    response = agent(prompt)

    # Convert response to dictionary
    try:
        extracted_data = json.loads(response)  # Replace eval with json.loads for safety
        return extracted_data
    except json.JSONDecodeError:
        raise ValueError("Failed to process natural language input.")


async def fetch_and_save_historical_data(stock_symbol: str, interval: str):
    """
    Fetch historical data from the stock API and save it as a CSV file.

    Args:
        stock_symbol (str): The stock symbol or function name.
        interval (str): The interval for the data (e.g., monthly, weekly).

    Returns:
        pd.DataFrame: The processed data.
    """
    url = f"{config.ALPHAVANTAGE_BASE_URL}?function={stock_symbol}&interval={interval}&apikey={config.ALPHAVANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Transform the data into a structured format
        structured_data = [{'date': entry['date'], 'value': entry['value']} for entry in data['data']]
        df = pd.DataFrame(structured_data)

        # Save to CSV
        df.to_csv(config.OUTPUT_CSV, index=False)
        print(f"Data saved to {config.OUTPUT_CSV}")

        return df
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        raise
    except KeyError as e:
        print(f"Error processing data: {e}")
        raise


def run_prediction_model(csv_path: str):
    """
    Run the prediction model on the saved CSV data.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        float: Predicted stock price for the target date.
    """
    # Load your prediction model
    model = PredictionModel("model/model_weights.pth")

    # Load data from CSV
    df = pd.read_csv(csv_path)

    # Process the data for prediction
    processed_data = model.preprocess_data(df)  # Assuming the model has a preprocess method

    # Run the prediction
    prediction = model.predict(processed_data)
    return prediction


async def execute_analysis_flow(input_text: str) -> AnalysisResponse:
    """
    Execute the complete analysis flow.

    Args:
        input_text (str): The natural language input from the user.

    Returns:
        AnalysisResponse: The response model containing the results.
    """
    try:
        # Step 1: Extract information using the agent
        extracted_data = process_natural_language(input_text)
        stock_symbol = extracted_data['stock_symbol']
        date_period = extracted_data['date_period']
        date_target = extracted_data['date_target']

        # Step 2: Fetch and save historical stock data
        df = await fetch_and_save_historical_data(stock_symbol, date_period)

        # Step 3: Run the prediction model
        predicted_price = run_prediction_model(config.OUTPUT_CSV)

        # Step 4: Create the final response
        return AnalysisResponse(
            generative_response=json.dumps(extracted_data),
            stock_symbol=stock_symbol,
            date_period=date_period,
            date_target=date_target,
            predicted_price=predicted_price,
        )
    except Exception as e:
        raise ValueError(f"Error in analysis flow: {e}")
