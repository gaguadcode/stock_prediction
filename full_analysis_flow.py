from app.graph_nodes.stock.stock_extractor_agent import StockDataExtractor
from app.graph_nodes.research.web_researcher_agent import StockResearchAgent
import asyncio

def process_and_analyze_stock(input_text: str, extractor_model="mistral", research_model="mistral"):
    """
    Processes natural language input to extract structured information,
    and runs the StockResearchAgent workflow to generate a detailed analysis report.

    Args:
        input_text (str): The natural language input describing the stock and prediction requirements.
        extractor_model (str): The model name for the StockDataExtractor. Default is "mistral".
        research_model (str): The model name for the StockResearchAgent. Default is "mistral".

    Returns:
        str: Final analysis report.
    """
    try:
        # Step 1: Extract structured data
        print("Initializing StockDataExtractor...")
        data_extractor = StockDataExtractor(model_name=extractor_model)
        print("Processing input text for stock data extraction...")
        agent_output = data_extractor.process_input(input_text)
        print("Extracted Agent Output:")
        print(agent_output)

        # Step 2: Initialize and run the research agent
        print("Initializing StockResearchAgent...")
        research_agent = StockResearchAgent(model_name=research_model)
        print("Starting Stock Research Workflow...")
        loop = asyncio.get_event_loop()
        final_response = loop.run_until_complete(research_agent.run(agent_output))

        print("\nFinal Analysis Report:")
        return final_response

    except Exception as e:
        return f"Error in stock processing and analysis workflow: {e}"

if __name__ == "__main__":
    # Input text in Spanish
    input_text = "quiero saber el price de IBM de manera mensual para enero de 2025"

    # Run the process_and_analyze_stock function with the provided input
    try:
        final_report = process_and_analyze_stock(input_text)
        print("\nFinal Report Output:")
        print(final_report)
    except Exception as e:
        print(f"Execution failed: {e}")
