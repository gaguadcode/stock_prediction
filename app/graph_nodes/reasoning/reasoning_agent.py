import json
from typing import Union, Dict
from app.utils.logger import get_logger
from app.utils.datatypes import UserInputString, ResearchOutput, ReasoningOutput
from langchain_ollama import OllamaLLM  # Import Ollama LLM from langchain-ollama
from app.utils.config import config
logger = get_logger("ReasoningNode")

class ReasoningNode:
    """
    Uses Ollama's DeepSeek model to generate reasoning for user input or research data.
    """

    def __init__(self):
        """
        Initializes the reasoning model inside the class.
        """
        logger.info("Initializing DeepSeek reasoning model via Ollama...")
        self.agent = OllamaLLM(model=config.REASONING_MODEL)  # Instantiates the Ollama model

    def construct_prompt(self, input_data: Dict) -> str:
        """
        Constructs a reasoning prompt for Ollama using all keys from the input dictionary.
        """
        prompt = (
            "Analyze the following input data and generate reasoning based on it:\n"
            f"{json.dumps(input_data, indent=2)}\n"
            "Explain the significance and implications of the provided information."
        )
        return prompt

    def generate_reasoning(self, input_data: Union[UserInputString, ResearchOutput]) -> ReasoningOutput:
        """
        Generates reasoning using Ollama's DeepSeek model.
        
        Args:
            input_data (Union[UserInputString, ResearchOutput]): Either user input 
            or research-related data.
        
        Returns:
            ReasoningOutput: AI-generated reasoning as structured output.
        """
        try:
            input_dict = input_data.model_dump()

            # Construct the reasoning prompt
            prompt = self.construct_prompt(input_dict)

            # Invoke DeepSeek model using Ollama
            logger.info("Invoking DeepSeek reasoning model...")
            response = self.agent.invoke(prompt)
            logger.info(f"Response from DeepSeek: {response}")

            # Parse response and return structured reasoning
            return ReasoningOutput(
                user_input=input_data.user_input,
                next_state=input_data.next_state,
                reasoning_output=response.strip()
            )

        except Exception as e:
            logger.error(f"Error in reasoning node: {e}")
            return ReasoningOutput(
                user_input=input_data.user_input,
                reasoning_output="Failed to generate reasoning."
            )
