import json
from typing import Union, Dict
from app.utils.llm_wrappers import LLMSelector  # ✅ Import LLM selector for dynamic LLM choice
from app.utils.logger import get_logger
from app.utils.datatypes import UserInputString, ResearchOutput, ReasoningOutput
from app.utils.config import config

# ✅ Initialize logger
logger = get_logger("ReasoningNode")

class ReasoningNode:
    """
    Uses an LLM (Google Gemini, OpenAI, or Ollama) to generate reasoning 
    for user input or research data.
    """

    def __init__(self):
        """
        Initializes the reasoning model dynamically using the LLM abstraction.
        """
        logger.info(f"Initializing ReasoningNode with LLM provider: {config.LLM_PROVIDER}")

        # ✅ Dynamically select LLM provider (Google Gemini, OpenAI, Ollama)
        self.agent = LLMSelector.get_llm(provider=config.LLM_PROVIDER, model_name=config.REASONING_MODEL)

        logger.info(f"ReasoningNode initialized with LLM: {config.LLM_PROVIDER}")

    def construct_prompt(self, input_data: Dict) -> str:
        """
        Constructs a reasoning prompt using all keys from the input dictionary.
        """
        prompt = (
            "Analyze the following input data (focusing in the user input key) and generate reasoning based on it:\n"
            f"{json.dumps(input_data, indent=2)}\n"
            "Explain the significance and implications of the provided information."
        )
        return prompt

    def generate_reasoning(self, input_data: Union[UserInputString, ResearchOutput]) -> ReasoningOutput:
        """
        Generates reasoning using the dynamically selected LLM.

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

            # Invoke the selected LLM model
            logger.info("Invoking LLM for reasoning...")
            response = self.agent.generate(prompt)  # ✅ Updated call to `.generate()`
            logger.info(f"Response from LLM: {response}")

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
                next_state=input_data.next_state,  # ✅ Ensure next_state persists
                reasoning_output="Failed to generate reasoning."
            )
