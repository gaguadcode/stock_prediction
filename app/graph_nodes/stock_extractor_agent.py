from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  

def initialize_langchain_agent():
    llm = OpenAI(temperature=0)  
    return llm

# LangChain agent for processing natural language
def process_natural_language(input_text: str):
    """
    Process the input natural language to extract stock symbol and date range.
    """
    # Initialize LangChain agent
    agent = initialize_langchain_agent()

    # Define a prompt to extract relevant data
    prompt_template = PromptTemplate(
        input_variables=["input_text"],
        template=(
            "Extract the stock symbol (company or comodity), prediction window (monthly, weekly, daily...), and target date of prediction from the following input:\n"
            "{input_text}\n"
            "Respond with JSON format: {{'stock_symbol': '...', 'date_period': '...', 'date_target':'...'}}"
        ),
    )

    prompt = prompt_template.format(input_text=input_text)
    response = agent(prompt)

    # Convert response to dictionary
    try:
        extracted_data = eval(response)  
        return extracted_data
    except Exception:
        raise ValueError("Failed to process natural language input.")

