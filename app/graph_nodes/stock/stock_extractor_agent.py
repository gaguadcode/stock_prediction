from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Import Ollama LLM from langchain-ollama
from app.utils.logger import get_logger
#from app.utils import validate_date_target  # Ensure this function is imported
from pydantic import BaseModel, Field
from typing import List, Literal
from app.utils.datatypes import UserInputString, StockPredictionRequest, EntityExtractOutput

import json

# Define input and output Pydantic model

class StockDataExtractor:
    """
    A LangGraph-compatible class to process natural language input using LangChain's OllamaLLM 
    and extract structured stock prediction information.
    """

    def __init__(self, model_name: str = "mistral:7b"):
        """
        Initializes the StockDataExtractor with the specified Ollama model.

        Args:
            model_name (str): The name of the Ollama model to use. Default is 'mistral'.
        """
        self.logger = get_logger(self.__class__.__name__)  # Initialize logger
        self.model_name = model_name
        self.logger.info(f"Initializing StockDataExtractor with model '{self.model_name}'")
        self.agent = self.initialize_llm()

    def initialize_llm(self) -> OllamaLLM:
        """
        Initialize the Ollama LLM agent.

        Returns:
            OllamaLLM: An instance of LangChain's OllamaLLM.
        """
        try:
            self.logger.info("Initializing Ollama LLM...")
            llm = OllamaLLM(model=self.model_name)
            self.logger.info("Ollama LLM initialized successfully.")
            return llm
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise ValueError(f"Failed to initialize Ollama LLM: {e}")

    def construct_prompt(self, input_text: str) -> str:
        """
        Constructs a prompt for the Ollama LLM to extract structured stock prediction data.

        Args:
            input_text (str): The natural language input.

        Returns:
            str: A formatted prompt ready for the LLM.
        """
        self.logger.info("Constructing prompt...")
        prompt_template = PromptTemplate(
            input_variables=["input_text"],
            template=(
                "Extract the stock symbol (company or commodity), prediction window (date_period) "
                "(TIME_SERIES_MONTHLY, TIME_SERIES_WEEKLY, TIME_SERIES_DAILY...), and target date or dates of prediction from the following input:\n"
                "{input_text}\n"
                "Respond strictly in JSON format:\n"
                "{{'stock_symbol': '...', 'date_period': '...', 'date_target':['YYYY-MM-DD',...]}}"
            ),
        )
        prompt = prompt_template.format(input_text=input_text)
        self.logger.debug(f"Prompt constructed: {prompt}")
        return prompt

    def process_input(self, input_data: UserInputString) -> EntityExtractOutput:
        """
        Processes the input natural language and returns a structured `StockPredictionRequest` 
        with `user_input` as the first key.

        Args:
            input_data (UserInputString): The input model containing user input.

        Returns:
            StockPredictionRequest: Extracted structured information with user input at the top.
        """
        try:
            self.logger.info("Processing input text...")

            # Extract raw user input
            input_text = input_data.user_input

            # Construct the prompt
            prompt = self.construct_prompt(input_text)

            # Generate response using Ollama LLM
            self.logger.info("Invoking Ollama LLM with the constructed prompt...")
            agent = self.initialize_llm()
            response = agent.invoke(prompt)
            self.logger.info(f"Response from LLM: {response}")

            # Parse JSON response
            extracted_data = json.loads(response.strip())  # Safer than eval
            self.logger.info("Input processed successfully.")
            self.logger.info(f"Extracted data: {extracted_data}")

            # Inject user_input at the beginning of the structured data
            extracted_data["user_input"] = input_text  
            stock_prediction = StockPredictionRequest(**extracted_data)
            # Validate and return structured output
            return EntityExtractOutput(
            user_input=input_text,
            stock_prediction=stock_prediction
        )

        except Exception as e:
            self.logger.error(f"Failed to process natural language input: {e}")
            raise ValueError(f"Failed to process natural language input: {e}")