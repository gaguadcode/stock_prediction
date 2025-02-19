import re
from langchain_ollama import OllamaLLM  # Import Ollama LLM from langchain-ollama
from langchain.tools import WikipediaLoader

class ResearcherNode:
    """
    Researcher Node that extracts one Wikipedia topic via LLM-based NER and fetches relevant summaries.
    The LLM also determines whether further reasoning is needed.
    """

    def __init__(self, model_name="mistral"):
        """
        Initializes the Researcher Node with an LLM model.
        """
        self.llm = OllamaLLM(model_name=model_name)
    
    def extract_topic(self, user_input):
        """
        Uses LLM to extract the **single most relevant** Wikipedia page topic.
        """
        prompt = f"Extract the **single most relevant** Wikipedia page topic from this query:\n\n{user_input}\n\n" \
                 "Return only the topic name, nothing else. If nothing is relevant, return 'None'."
        response = self.llm.predict(prompt).strip()
        
        if response.lower() == "none" or not response:
            return None
        return response

    def fetch_wikipedia_data(self, topic):
        """
        Fetch Wikipedia summary for the extracted topic.
        """
        if not topic:
            return "No specific Wikipedia topic found, proceeding with general reasoning."

        try:
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            docs = loader.load()
            summary = docs[0].metadata.get("summary", "No summary available.") if docs else "No Wikipedia page found."
            return summary
        except Exception as e:
            return f"Error fetching Wikipedia data: {e}"

    def should_call_reasoning(self, research_summary, user_input):
        """
        Uses LLM to determine whether further reasoning is required.
        """
        prompt = f"""
        Given the following research summary:
        "{research_summary}"

        And the user's original request:
        "{user_input}"

        Determine whether further reasoning is needed.
        Respond only with "Yes" or "No".
        """
        response = self.llm.predict(prompt).strip().lower()
        return response == "yes"

    def researcher(self, inputs):
        """
        Executes the researcher node workflow.
        """
        user_input = inputs["input"]
        
        # Step 1: Extract the single best Wikipedia topic
        extracted_topic = self.extract_topic(user_input)
        
        # Step 2: Fetch Wikipedia summary if a topic was found
        research_summary = self.fetch_wikipedia_data(extracted_topic)
        
        # Step 3: Ask LLM if reasoning is required
        call_reasoning = self.should_call_reasoning(research_summary, user_input)
        
        # Step 4: Output format
        response = {
            "research_result": research_summary,
            "call_reasoning": call_reasoning  # Now conditionally set based on LLM output
        }
        
        return response
