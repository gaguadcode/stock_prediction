from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM
from app.utils.logger import get_logger
from app.utils.datatypes import (
    UserInputString, ResearchOutput, ReasoningOutput, 
    EntityExtractOutput, DataFetchOutput, FinalPredictionState, 
    StockPredictionOutput
)
from app.graph_nodes.stock.stock_extractor_agent import StockDataExtractor
from app.graph_nodes.stock.stock_fetch_data import HistoricalDataFetcher
from app.graph_nodes.stock.model_training import StockDataTrainer
from app.graph_nodes.stock.prediction import StockPredictor
from app.graph_nodes.research.web_researcher_agent import ResearcherNode
from app.graph_nodes.reasoning.reasoning_agent import ReasoningNode
from app.graph_nodes.main_router import MainRouterNode

# ✅ Initialize Logger
logger = get_logger("MainRouterNode")

# ✅ **Fix: Initialize with `UserInputString` as starting state**
graph = StateGraph(UserInputString)

# ✅ **Step 1: Router Node**
def router_node(state: UserInputString) -> UserInputString:
    """
    Determines the next step and returns a `RouterOutput` object.
    """
    router = MainRouterNode()
    return router.determine_route(state)  # ✅ Returns `RouterOutput`

graph.add_node("router", router_node)

# ✅ **Step 2: Stock Prediction Pipeline**
def entity_extraction_node(state: UserInputString) -> EntityExtractOutput:
    logger.info("Running Entity Extraction Node...")
    extractor = StockDataExtractor()
    extracted_data = extractor.process_input(state)  # ✅ Extract `UserInputString`
    return extracted_data

graph.add_node("entity_extraction", entity_extraction_node)

def data_fetching_node(state: EntityExtractOutput) -> DataFetchOutput:
    logger.info("Running Data Fetching Node...")
    fetcher = HistoricalDataFetcher()
    data_fetch_output = fetcher.fetch_and_store_historical_data(state)
    return data_fetch_output

graph.add_node("data_fetching", data_fetching_node)

def training_node(state: DataFetchOutput) -> FinalPredictionState:
    logger.info("Running Training Node...")
    trainer = StockDataTrainer()
    final_state = trainer.execute_training(state)
    return final_state

graph.add_node("training", training_node)

def prediction_node(state: FinalPredictionState) -> StockPredictionOutput:
    logger.info("Running Prediction Node...")
    predictor = StockPredictor(final_state=state)
    prediction_output = predictor.make_predictions(state)
    return prediction_output

graph.add_node("prediction", prediction_node)

# ✅ **Step 3: Researcher Node**
def researcher_node(state: UserInputString) -> ResearchOutput:
    logger.info("Executing Researcher Node...")
    researcher = ResearcherNode()
    research_output = researcher.researcher(state)  # ✅ Extract `UserInputString`
    return research_output

graph.add_node("researcher", researcher_node)

# ✅ **Step 4: Reasoning Node**
def reasoning_node(state: UserInputString) -> ReasoningOutput:
    logger.info("Executing Reasoning Node...")
    reasoner = ReasoningNode()
    reasoning_output = reasoner.generate_reasoning(state)  # ✅ Extract `UserInputString`
    return reasoning_output

graph.add_node("reasoning", reasoning_node)

# ✅ **Fix: Ensure Router Output is Used for Conditional Routing**
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    lambda state: state.next_state,  # ✅ Correctly extract route string
    {
        "entity_extraction": "entity_extraction",
        "researcher": "researcher",
        "reasoning": "reasoning"
    }
)

# ✅ **Stock Prediction Flow**
graph.add_edge("entity_extraction", "data_fetching")
graph.add_edge("data_fetching", "training")
graph.add_edge("training", "prediction")
graph.add_edge("prediction", END)

# ✅ **Research & Reasoning Endpoints**
graph.add_edge("researcher", END)
graph.add_edge("reasoning", END)

# ✅ **Compile the Graph**
workflow = graph.compile()

# ✅ **Fix: Start with `UserInputString`, Expect `RouterOutput` Transition**
test = UserInputString(user_input="reasoning about quantum computing", next_state="")  # Stock Prediction

output = workflow.invoke(test)
logger.info(f"Workflow Output: {output}\n")
