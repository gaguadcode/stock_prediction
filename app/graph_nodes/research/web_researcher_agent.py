import re
from app.utils.llm_wrappers import LLMSelector  # ✅ Import LLM selector
from langchain_community.document_loaders.wikipedia import WikipediaLoader  # ✅ Corrected Import
from app.utils.logger import get_logger
from app.utils.datatypes import UserInputString, ResearchOutput
from app.utils.config import config

# ✅ Initialize logger
logger = get_logger("ResearcherNode")

class ResearcherNode:
    """
    Researcher Node that extracts one Wikipedia topic via LLM-based NER and fetches relevant summaries.
    The LLM also determines whether further reasoning is needed.
    """

    def __init__(self):
        """
        Initializes the Researcher Node with a dynamically selected LLM.
        """
        logger.info(f"Initializing ResearcherNode with LLM provider: {config.LLM_PROVIDER}")

        # ✅ Dynamically select LLM provider (Google Gemini, OpenAI, Ollama)
        self.llm = LLMSelector.get_llm(provider=config.LLM_PROVIDER, model_name=config.RESEARCH_MODEL)

        logger.info(f"ResearcherNode initialized with LLM: {config.LLM_PROVIDER}")

    def extract_topic(self, user_input: str) -> str:
        """
        Uses LLM to extract the **single most relevant** Wikipedia page topic.
        """
        logger.info(f"Extracting topic from user input: {user_input}")

        prompt = f"""
        Extract the **single most relevant** Wikipedia page topic from this query:
        
        "{user_input}"
        
        - Return **only the topic name**, nothing else.
        - If no topic is relevant, return **None**.
        """
        
        try:
            response = self.llm.generate(prompt).strip()
            logger.info(f"Extracted topic: {response}")

            if response.lower() == "none" or not response:
                logger.warning("No relevant Wikipedia topic found.")
                return None
            
            return response
        except Exception as e:
            logger.error(f"Error extracting topic: {e}")
            return None

    def fetch_wikipedia_data(self, topic: str) -> str:
        """
        Fetch Wikipedia summary for the extracted topic.
        """
        if not topic:
            logger.info("No topic extracted, proceeding with general reasoning.")
            return "No specific Wikipedia topic found, proceeding with general reasoning."

        try:
            logger.info(f"Fetching Wikipedia summary for topic: {topic}")
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            docs = loader.load()

            if docs:
                summary = docs[0].metadata.get("summary", "No summary available.")
                logger.info("Successfully fetched Wikipedia summary.")
            else:
                summary = "No Wikipedia page found."
                logger.warning("No Wikipedia page found for the topic.")

            return summary
        except Exception as e:
            logger.error(f"Error fetching Wikipedia data: {e}")
            return f"Error fetching Wikipedia data: {e}"

    def researcher(self, inputs: UserInputString) -> ResearchOutput:
        """
        Executes the researcher node workflow.
        """
        user_input_str = inputs.user_input  # Extract the raw string from UserInputString
        logger.info(f"Running researcher node for input: {user_input_str}")

        # Step 1: Extract the single best Wikipedia topic
        extracted_topic = self.extract_topic(user_input_str)
        
        # Step 2: Fetch Wikipedia summary if a topic was found
        research_summary = self.fetch_wikipedia_data(extracted_topic)
        
        # Step 3: Return output in structured ResearchOutput
        logger.info("Research node execution complete.")
        return ResearchOutput(
            user_input=user_input_str,  # Retain original user input
            research_output=research_summary
        )
