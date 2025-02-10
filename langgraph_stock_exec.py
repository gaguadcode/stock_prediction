from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Dict
from app.utils.logger import get_logger
from app.utils.datatypes import (
    UserInputString,
    EntityExtractOutput,
    DataFetchOutput,
    FinalPredictionState,
    StockPredictionRequest,
    StockPredictionOutput
)
from app.graph_nodes.stock.stock_extractor_agent import StockDataExtractor
from app.graph_nodes.stock.stock_fetch_data import HistoricalDataFetcher
from app.graph_nodes.stock.model_training import StockDataTrainer
from app.graph_nodes.stock.prediction import StockPredictor

fake_data_fetch_output = DataFetchOutput(
    entity_extract=EntityExtractOutput(
        user_input="quiero saber el precio de IBM de manera mensual para enero de 2025",
        stock_prediction=StockPredictionRequest(
            stock_symbol="IBM",
            date_period="TIME_SERIES_MONTHLY",
            date_target=["2025-01-01"]
        )
    ),
    database_url="postgresql://gustavo:password@localhost/ibm_time_series_monthly"
)

# **✅ Initialize LangGraph workflow**
graph = StateGraph(UserInputString)

# **✅ Logger setup**
logger = get_logger("StockPredictionWorkflow")
'''
# **✅ Step 1: Entity Extraction Node**
def entity_extraction_node(state: UserInputString) -> EntityExtractOutput:
    logger.info("Running Entity Extraction Node...")
    extractor = StockDataExtractor()
    extracted_data = extractor.process_input(state)

    return extracted_data

graph.add_node("entity_extraction", entity_extraction_node)

# **✅ Step 2: Data Fetching Node**
def data_fetching_node(state: EntityExtractOutput) -> DataFetchOutput:
    logger.info("Running Data Fetching Node...")
    fetcher = HistoricalDataFetcher()
    data_fetch_output = fetcher.fetch_and_store_historical_data(
        state)
    
    return data_fetch_output

graph.add_node("data_fetching", data_fetching_node)
'''
# **✅ Step 3: Training Node**
def training_node(state: DataFetchOutput) -> FinalPredictionState:
    logger.info("Running Training Node...")
    trainer = StockDataTrainer()
    final_state = trainer.execute_training(DataFetchOutput(state))

    return final_state

graph.add_node("training", training_node)

# **✅ Step 4: Prediction Node**
def prediction_node(state: FinalPredictionState) -> StockPredictionOutput:
    logger.info("Running Prediction Node...")
    predictor = StockPredictor(FinalPredictionState(**state["final_state"]))

    stock_request = state["final_state"]["entity_extract"]["entity_extract"]["stock_prediction"]
    stock_request_obj = StockPredictionRequest(**stock_request)

    prediction_output = predictor.make_predictions(stock_request_obj)

    return prediction_output

graph.add_node("prediction", prediction_node)

# **✅ Define Execution Order**
#graph.add_edge(START, "entity_extraction")
#graph.add_edge("entity_extraction", "data_fetching")
graph.add_edge(START, "training")
graph.add_edge("training", "prediction")
graph.add_edge("prediction", END)

# **✅ Compile the Graph**
workflow = graph.compile()

# **✅ Running the Workflow**
state = workflow.invoke({
    "user_input": "quiero saber el precio de IBM de manera mensual para enero de 2025"
})

print(state)  # Output the final workflow state
