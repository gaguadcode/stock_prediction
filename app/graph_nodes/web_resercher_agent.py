from langchain_community.document_loaders import WikipediaLoader
from langchain.llms import OpenAI

def fetch_wikipedia_data(stock_name):
    try:
        loader = WikipediaLoader(query=stock_name, load_max_docs=1)
        docs = loader.load()
        if docs:
            return docs[0].metadata.get("summary", "No summary available.")
        else:
            return "No Wikipedia page found for this stock."
    except Exception as e:
        return f"Error fetching data: {e}"


def generate_elaborate_response(stock_name, wikipedia_summary):
    llm = OpenAI(temperature=0.7)
    prompt = f"""
    You are an expert financial researcher. Based on the following Wikipedia summary for {stock_name}, provide a detailed explanation about the company, its significance, and historical context:
    
    Wikipedia Summary:
    {wikipedia_summary}
    
    Detailed response:
    """
    return llm(prompt)

def stock_research_agent(stock_name):
    # Step 1: Fetch Wikipedia data
    wikipedia_summary = fetch_wikipedia_data(stock_name)
    
    # Step 2: Generate an elaborate response
    if "No Wikipedia page found" not in wikipedia_summary:
        response = generate_elaborate_response(stock_name, wikipedia_summary)
    else:
        response = wikipedia_summary  # Return error message directly if no data found
    
    return response
