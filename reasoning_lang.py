from langgraph.graph import StateGraph, START, END
from app.utils.logger import get_logger
from app.utils.datatypes import UserInputString, ResearchOutput, ReasoningOutput
from app.graph_nodes.reasoning.reasoning_agent import ReasoningNode  # Ensure correct import

# ✅ Initialize LangGraph workflow
graph = StateGraph(ResearchOutput)

# ✅ Logger setup
logger = get_logger("ReasoningGraphTest")

# ✅ Step 1: Reasoning Node
def reasoning_node(state: ResearchOutput) -> ReasoningOutput:
    """
    Calls the ReasoningNode to generate reasoning based on the provided ResearchOutput.
    """
    logger.info("Executing Reasoning Node Test...")
    reasoner = ReasoningNode()
    
    reasoning_result = reasoner.generate_reasoning(state)

    logger.info(f"Reasoning Output: {reasoning_result.reasoning_output}")
    
    return reasoning_result

graph.add_node("reasoning", reasoning_node)

# ✅ Define Execution Order
graph.add_edge(START, "reasoning")
graph.add_edge("reasoning", END)

# ✅ Compile the Graph
workflow = graph.compile()

# ✅ Running the Workflow (Test Case)
test_state = ResearchOutput(
    user_input="What impact does AI have on the job market?",
    research_output="AI is transforming industries by automating tasks, augmenting decision-making, and creating new job roles."
)

state = workflow.invoke(test_state)

# ✅ Output the final workflow state
print(state)
