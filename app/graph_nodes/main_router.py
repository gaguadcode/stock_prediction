from langchain_ollama import OllamaLLM
from app.utils.logger import get_logger
from app.utils.datatypes import UserInputString
import re
from app.utils.config import config
# ✅ Initialize Logger
logger = get_logger("MainRouterNode")

class MainRouterNode:
    """
    Routes user input to either Stock Prediction, Research, or Reasoning using both
    a keyword-based approach and an LLM validation step.
    """

    def __init__(self):
        """
        Initializes components needed for routing.
        """
        self.stock_keywords = {"stock", "price", "share", "market", "invest", "trading"}
        self.research_keywords = {"what is", "explain", "define", "history of"}
        self.reasoning_keywords = {"impact", "analyze", "predict", "reasoning", "effect of"}

        # ✅ Initialize DeepSeek LLM for Routing Validation
        self.llm = OllamaLLM(model=config.ROUTING_MODEL)

        logger.info("MainRouterNode initialized.")

    def keyword_based_routing(self, user_input: str) -> str:
        """
        Fast keyword-based routing.
        """
        lower_input = user_input.lower()

        if any(keyword in lower_input for keyword in self.stock_keywords):
            return "entity_extraction"  # ✅ Matches the actual graph node name
        if any(keyword in lower_input for keyword in self.research_keywords):
            return "researcher"
        if any(keyword in lower_input for keyword in self.reasoning_keywords):
            return "reasoning"

        return "undecided"  # If keywords don't match, LLM makes the final call

    def llm_based_routing(self, user_input: str) -> str:
        """
        Uses DeepSeek LLM to analyze user input and determine the appropriate route.
        Ensures the LLM only returns a valid route and removes unwanted text.
        """
        logger.info("Calling DeepSeek LLM for routing decision...")

        prompt = f"""
        Given the following user input:

        "{user_input}"

        Determine the most suitable processing route:
        - If it is about stock prices, financial markets, or investing, return: **entity_extraction**
        - If it is about general knowledge, Wikipedia topics, or explanations, return: **researcher**
        - If it requires analysis, reasoning, critical thinking, or predictions, return: **reasoning**

        **Respond only with one of these exact words: "entity_extraction", "researcher", or "reasoning". No explanation, no reasoning, just the word.**
        """

        response = self.llm.invoke(prompt).strip().lower()

        # ✅ Extract only the first valid response (preventing full explanations)
        match = re.search(r"\b(entity_extraction|researcher|reasoning)\b", response)

        if match:
            return match.group(1)  # ✅ Extracts and returns only the valid route

        logger.warning(f"LLM returned an unexpected response: {response}. Defaulting to 'reasoning'.")
        return "reasoning"  # Default fallback
    
    def determine_route(self, initial_state: UserInputString) -> UserInputString:
        """
        Uses a hybrid approach:
        1. **Keyword Matching**
        2. **DeepSeek LLM Validation**
        
        ✅ Returns a dictionary with the route AND original UserInputString.
        """
        fast_route = self.keyword_based_routing(initial_state.user_input)
        logger.info(f"Keyword-based Routing Suggests: {fast_route}")

        if fast_route == "undecided":
            final_route = self.llm_based_routing(initial_state.user_input)
        else:
            final_route = fast_route

        logger.info(f"Final Route Selected: {final_route}")
        # ✅ Return BOTH the routing decision and UserInputString
        return UserInputString(next_state=final_route, user_input=initial_state.user_input) 
