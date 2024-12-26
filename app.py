import streamlit as st
from app.graph_nodes.stock_extractor_agent import StockDataExtractor
from app.graph_nodes.web_researcher_agent import StockResearchAgent
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
        st.info("Initializing StockDataExtractor...")
        data_extractor = StockDataExtractor(model_name=extractor_model)
        st.info("Processing input text for stock data extraction...")
        agent_output = data_extractor.process_input(input_text)
        st.success("Extracted Agent Output:")
        st.json(agent_output)

        # Step 2: Initialize and run the research agent
        st.info("Initializing StockResearchAgent...")
        research_agent = StockResearchAgent(model_name=research_model)
        st.info("Starting Stock Research Workflow...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_response = loop.run_until_complete(research_agent.run(agent_output))

        st.success("Final Analysis Report:")
        return final_response

    except Exception as e:
        st.error(f"Error in stock processing and analysis workflow: {e}")
        return str(e)

# Streamlit app layout
st.title("Stock Research and Analysis")
st.write("This app processes natural language input to analyze stock data and predictions.")

# Input section
input_text = st.text_area("Enter your query about stock prices or predictions:", 
                          placeholder="E.g., 'quiero saber el precio de IBM de manera mensual para enero de 2025'")

# Model selection
extractor_model = st.selectbox("Select Stock Data Extractor Model:", ["mistral", "default_model"])
research_model = st.selectbox("Select Stock Researcher Agent Model:", ["mistral", "default_model"])

# Process the input and display results
if st.button("Analyze Stock"):
    if input_text.strip():
        st.info("Processing your request...")
        final_report = process_and_analyze_stock(input_text, extractor_model, research_model)
        st.subheader("Analysis Report")
        st.write(final_report)
    else:
        st.warning("Please enter a query to analyze.")
