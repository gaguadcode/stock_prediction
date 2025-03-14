import json
from typing import Dict
from app.utils.llm_wrappers import LLMSelector  # ✅ Import LLM selector for dynamic LLM choice
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState  # ✅ Unified WorkflowState
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

        logger.info(f"✅ ReasoningNode initialized with LLM: {config.LLM_PROVIDER}")

    def construct_prompt(self, state: Dict) -> str:
        """
        Constructs a reasoning prompt using all keys from the current workflow state.
        """
        prompt = (
            "Analyze the following user input and associated data, and generate reasoning based on it:\n"
            f"{json.dumps(state, indent=2)}\n"
            "Explain the significance and implications of the provided information. avoid mentioning the format of the json, because its irrelevant to the reasoning. alsofocus in the user input key and answer it."
        )
        return prompt

    def generate_reasoning(self, state: WorkflowState) -> WorkflowState:
        """
        Generates reasoning using the dynamically selected LLM.
        Updates only `None` values in WorkflowState.
        """
        try:
            state_dict = state.model_dump()

            # ✅ Construct reasoning prompt based on the entire state
            prompt = self.construct_prompt(state_dict)

            # ✅ Invoke LLM model for reasoning
            logger.info("🧠 Invoking LLM for reasoning...")
            response = self.agent.generate(prompt)
            logger.info(f"📖 Response from LLM: {response}")

            # ✅ Update only if `reasoning_output` is None
            updated_data = {
                "reasoning_output": response.strip() if state.reasoning_output is None else state.reasoning_output
            }

            # ✅ Merge updated data while preserving prior state
            return state.model_copy(update=updated_data)

        except Exception as e:
            logger.error(f"❌ Error in reasoning node: {e}")

            return state.model_copy(update={
                "reasoning_output": state.reasoning_output or "⚠️ Failed to generate reasoning."
            })