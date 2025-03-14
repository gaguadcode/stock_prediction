import json
import re
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState
from app.utils.llm_wrappers import LLMSelector
from app.utils.config import config

class StockDataExtractor:
    """
    Processes natural language input to extract structured stock prediction data.
    """

    def __init__(self):
        """
        Initializes the StockDataExtractor with the appropriate LLM provider.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"🔄 Initializing StockDataExtractor with model '{config.ENTITY_EXTRACTION_MODEL}'")
        
        # ✅ Dynamically select LLM (Google Gemini, OpenAI, Ollama)
        self.agent = LLMSelector.get_llm(provider=config.LLM_PROVIDER, model_name=config.ENTITY_EXTRACTION_MODEL)

    def clean_python_dict_response(self, response: str) -> dict:
        """
        Cleans and extracts a valid Python dictionary from the LLM response.
        Handles cases where the response contains Markdown-style formatting or additional text.
        """
        try:
            self.logger.info(f"📝 Raw response before parsing: {response}")

            # ✅ Remove Markdown-style formatting (` ```python ... ``` ` or ` ``` `)
            response = re.sub(r"```python\n?|```", "", response).strip()

            # ✅ Convert the string response into a valid Python dictionary
            extracted_data = eval(response)  # ⚠️ Use eval ONLY because we control the prompt

            # ✅ Ensure all required keys exist
            required_keys = {"stock_symbol", "date_period", "date_target"}
            if not required_keys.issubset(extracted_data.keys()):
                raise ValueError("❌ Missing expected keys in Python dictionary response.")

            return extracted_data

        except Exception as e:
            self.logger.error(f"❌ Python dictionary parsing failed: {e}")
            raise ValueError(f"Invalid Python dictionary format: {response}")

    def construct_prompt(self, input_text: str) -> str:
        """
        Constructs a structured prompt for the LLM to return a Python dictionary.
        """
        self.logger.info("🛠 Constructing prompt for entity extraction...")
        prompt = (
            "Extract the stock symbol (company or commodity), prediction window (date_period) "
            "(TIME_SERIES_MONTHLY, TIME_SERIES_WEEKLY, TIME_SERIES_DAILY), and target date(s) "
            "from the following user input:\n"
            f"{input_text}\n"
            "Respond strictly in Python dictionary format (no Markdown, no extra text):\n"
            "{'stock_symbol': '...', 'date_period': '...', 'date_target': ['YYYY-MM-DD', ...]}"
        )
        return prompt

    def process_input(self, state: WorkflowState) -> WorkflowState:
        """
        Processes user input and extracts structured stock prediction data.
        Updates only initially `None` values while preserving previously set values.
        """
        try:
            self.logger.info(f"🔍 Processing input text: {state.user_input}")

            # ✅ Construct the prompt
            prompt = self.construct_prompt(state.user_input)

            # ✅ Invoke LLM (Google Gemini, OpenAI, Ollama)
            self.logger.info("🤖 Invoking LLM for structured extraction...")
            response = self.agent.generate(prompt)
            self.logger.info(f"📜 Response from LLM: {response}")

            # ✅ Clean and parse the Python dictionary response
            extracted_data = self.clean_python_dict_response(response)
            self.logger.info(f"✅ Extracted structured data: {extracted_data}")

            # ✅ Update only `None` values while preserving existing data
            updated_data = {
                key: value if getattr(state, key) is None else getattr(state, key)
                for key, value in extracted_data.items()
            }

            # ✅ Create a new state with updated values
            return state.model_copy(update=updated_data)

        except Exception as e:
            self.logger.error(f"❌ Failed to process input: {e}")
            raise ValueError(f"Failed to process input: {e}")
