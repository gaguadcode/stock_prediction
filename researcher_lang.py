from langgraph.graph import StateGraph, START, END
from app.utils.logger import get_logger
from app.utils.datatypes import UserInputString, ResearchOutput
from app.graph_nodes.research.web_researcher_agent import ResearcherNode  # Ensure correct import

# ✅ Initialize LangGraph workflow
graph = StateGraph(UserInputString)

# ✅ Logger setup
logger = get_logger("ResearcherGraphTest")

# ✅ Step 1: Researcher Node
def researcher_node(state: UserInputString) -> ResearchOutput:
    """
    Calls the ResearcherNode to extract Wikipedia data based on user input.
    """
    logger.info("Executing Researcher Node Test...")
    researcher = ResearcherNode()
    
    research_result = researcher.researcher(state)

    logger.info(f"Research Output: {research_result.research_output}")
    
    return research_result

graph.add_node("research", researcher_node)

# ✅ Define Execution Order
graph.add_edge(START, "research")
graph.add_edge("research", END)

# ✅ Compile the Graph
workflow = graph.compile()

# ✅ Running the Workflow (Test Case)
test_state = UserInputString(user_input="Tell me about Quantum Computing")

state = workflow.invoke(test_state)

# ✅ Output the final workflow state
print(state)
