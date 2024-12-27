from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Import Ollama LLM from langchain-ollama
from app.utils.logger import get_logger


class StockDataExtractor:
    """
    A class to process natural language input using LangChain's OllamaLLM and extract structured information.

    Methods:
        initialize_llm: Initializes the Ollama LLM agent.
        construct_prompt: Constructs a structured prompt for processing input.
        process_input: Processes natural language input and extracts structured information.
    """

    def __init__(self, model_name: str = "mistral"):
        """
        Initializes the NaturalLanguageProcessor with the specified Ollama model.

        Args:
            model_name (str): The name of the Ollama model to use. Default is 'mistral'.
        """
        self.logger = get_logger(self.__class__.__name__)  # Initialize a logger for this class
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
        Constructs a prompt for the Ollama LLM to extract structured data.

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

    def process_input(self, input_text: str) -> dict:
        """
        Processes the input natural language to extract structured data.

        Args:
            input_text (str): The natural language input.

        Returns:
            dict: Extracted structured information including stock symbol, date period, and target date.
        """
        try:
            self.logger.info("Processing input text...")
            
            # Construct the prompt
            prompt = self.construct_prompt(input_text)

            # Generate response using Ollama LLM
            self.logger.info("Invoking Ollama LLM with the constructed prompt...")
            response = self.agent.invoke(prompt)
            self.logger.debug(f"Response from LLM: {response}")

            # Parse response into a dictionary
            extracted_data = eval(response.strip())  # Parse string into dictionary
            self.logger.info("Input processed successfully.")
            self.logger.debug(f"Extracted data: {extracted_data}")
            return extracted_data
        except Exception as e:
            self.logger.error(f"Failed to process natural language input: {e}")
            raise ValueError(f"Failed to process natural language input: {e}")
