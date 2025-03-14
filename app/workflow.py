from langgraph.graph import StateGraph, START, END
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState
from app.graph_nodes.stock.stock_extractor_agent import StockDataExtractor
from app.graph_nodes.stock.stock_fetch_data import HistoricalDataFetcher
from app.graph_nodes.stock.model_training import StockDataTrainer
from app.graph_nodes.stock.prediction import StockPredictor
from app.graph_nodes.research.web_researcher_agent import ResearcherNode
from app.graph_nodes.reasoning.reasoning_agent import ReasoningNode
from app.graph_nodes.main_router import MainRouterNode

logger = get_logger("Workflow")

def create_workflow():
    """
    Initializes and returns the compiled AI workflow.
    """
    graph = StateGraph(WorkflowState)

    # ✅ **Step 1: Router Node**
    def router_node(state: WorkflowState) -> WorkflowState:
        """
        Determines the next step in the workflow.
        """
        router = MainRouterNode()
        result = router.determine_route(state)  # ✅ Already returns `WorkflowState`

        logger.info("📍 Router Output: %s", result.model_dump())

        return result  # ✅ Directly return result

    graph.add_node("router", router_node)

    # ✅ **Step 2: Stock Prediction Pipeline**
    def entity_extraction_node(state: WorkflowState) -> WorkflowState:
        """
        Extracts stock data entities from user input.
        """
        logger.info("🔍 Running Entity Extraction...")
        extractor = StockDataExtractor()
        extracted_data = extractor.process_input(state)  # ✅ Returns `WorkflowState`

        logger.info("📊 Entity Extraction Output: %s", extracted_data.model_dump())

        return extracted_data  # ✅ Directly return modified state

    graph.add_node("entity_extraction", entity_extraction_node)

    def data_fetching_node(state: WorkflowState) -> WorkflowState:
        """
        Fetches historical stock data and stores it in a database.
        """
        logger.info("📡 Running Data Fetching Node...")
        fetcher = HistoricalDataFetcher()

        # ✅ Fetch and return data
        data_fetch_output = fetcher.fetch_and_store_historical_data(state)  # ✅ Returns `WorkflowState`

        logger.info("🗄️ Data Fetch Output: %s", data_fetch_output.model_dump())

        return data_fetch_output  # ✅ Directly return modified state

    graph.add_node("data_fetching", data_fetching_node)

    def training_node(state: WorkflowState) -> WorkflowState:
        """
        Trains a machine learning model for stock price prediction.
        """
        logger.info("🛠️ Training Model...")
        trainer = StockDataTrainer()
        final_state = trainer.execute_training(state)  # ✅ Returns `WorkflowState`

        logger.info("📈 Model Training Output: %s", final_state.model_dump())

        return final_state  # ✅ Directly return modified state

    graph.add_node("training", training_node)

    def prediction_node(state: WorkflowState) -> WorkflowState:
        """
        Uses a trained model to predict stock prices.
        """
        logger.info("🔮 Making Predictions...")

        # ✅ Correct argument name when creating `StockPredictor`
        predictor = StockPredictor(state=state)  

        # ✅ Call `make_predictions()` without passing `state` again
        prediction_output = predictor.make_predictions()  

        # 🔍 Ensure `prediction_output` is of the correct type before logging
        if not isinstance(prediction_output, WorkflowState):
            raise TypeError(f"❌ prediction_output is {type(prediction_output)}, expected WorkflowState")

        logger.info("📊 Prediction Output: %s", prediction_output.model_dump())

        return prediction_output


    graph.add_node("prediction", prediction_node)

    # ✅ **Step 3: Researcher Node**
    def researcher_node(state: WorkflowState) -> WorkflowState:
        """
        Conducts research and retrieves relevant information.
        """
        logger.info("🔍 Researching Topic...")
        researcher = ResearcherNode()
        research_output = researcher.researcher(state)  # ✅ Returns `WorkflowState`

        logger.info("📖 Research Output: %s", research_output.model_dump())

        return research_output  # ✅ Directly return modified state

    graph.add_node("researcher", researcher_node)

    # ✅ **Step 4: Reasoning Node**
    def reasoning_node(state: WorkflowState) -> WorkflowState:
        """
        Generates reasoning for a given topic.
        """
        logger.info("💡 Generating Reasoning...")
        reasoner = ReasoningNode()
        reasoning_output = reasoner.generate_reasoning(state)  # ✅ Returns `WorkflowState`

        logger.info("🧠 Reasoning Output: %s", reasoning_output.model_dump())

        return reasoning_output  # ✅ Directly return modified state

    graph.add_node("reasoning", reasoning_node)

    # ✅ **Define Execution Order**
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state.next_state,  
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
    graph.add_edge("prediction", END)  # ✅ Ensure final state comes from prediction node

    # ✅ **Research & Reasoning Endpoints**
    graph.add_edge("researcher", END)
    graph.add_edge("reasoning", END)

    return graph.compile()
