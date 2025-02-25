import streamlit as st
import time
from langgraph.graph import StateGraph, START, END
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

# 🎨 Streamlit UI Enhancements
st.set_page_config(page_title="Stock & Research AI Workflow", layout="wide")
st.title("📊 AI Workflow Execution")

# ✅ Initialize Logger
logger = get_logger("WorkflowApp")

# ✅ **Step 1: User Input**
st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter your query:", "quiero saber el precio de IBM de manera mensual para enero de 2025")

if st.sidebar.button("Run Workflow"):
    st.subheader("🚀 Workflow Execution Progress")
    
    # ✅ **Initialize LangGraph Workflow**
    graph = StateGraph(UserInputString)
    
    # ✅ **Step 1: Router Node**
    def router_node(state: UserInputString) -> UserInputString:
        router = MainRouterNode()
        result = router.determine_route(state)
        st.write("📍 **Router Output:**", result.dict())  # ✅ Log state visually
        time.sleep(1)
        return result

    graph.add_node("router", router_node)

    # ✅ **Step 2: Stock Prediction Pipeline**
    def entity_extraction_node(state: UserInputString) -> EntityExtractOutput:
        st.write("🔍 **Running Entity Extraction...**")
        extractor = StockDataExtractor()
        extracted_data = extractor.process_input(state)
        st.write("📊 **Entity Extraction Output:**", extracted_data.dict())
        time.sleep(1)
        return extracted_data

    graph.add_node("entity_extraction", entity_extraction_node)

    def data_fetching_node(state: EntityExtractOutput) -> DataFetchOutput:
        st.write("📡 **Fetching Historical Data...**")
        fetcher = HistoricalDataFetcher()
        data_fetch_output = fetcher.fetch_and_store_historical_data(state)
        st.write("🗄️ **Data Fetch Output:**", data_fetch_output.dict())
        time.sleep(1)
        return data_fetch_output

    graph.add_node("data_fetching", data_fetching_node)

    def training_node(state: DataFetchOutput) -> FinalPredictionState:
        st.write("🛠️ **Training Model...**")
        trainer = StockDataTrainer()
        final_state = trainer.execute_training(state)
        st.write("📈 **Model Training Output:**", final_state.dict())
        time.sleep(1)
        return final_state

    graph.add_node("training", training_node)

    def prediction_node(state: FinalPredictionState) -> StockPredictionOutput:
        st.write("🔮 **Making Predictions...**")
        predictor = StockPredictor(final_state=state)
        prediction_output = predictor.make_predictions(state)
        st.write("📊 **Prediction Output:**", prediction_output.dict())
        time.sleep(1)
        return prediction_output

    graph.add_node("prediction", prediction_node)

    # ✅ **Step 3: Researcher Node**
    def researcher_node(state: UserInputString) -> ResearchOutput:
        st.write("🔍 **Researching Topic...**")
        researcher = ResearcherNode()
        research_output = researcher.researcher(state)
        st.write("📖 **Research Output:**", research_output.dict())
        time.sleep(1)
        return research_output

    graph.add_node("researcher", researcher_node)

    # ✅ **Step 4: Reasoning Node**
    def reasoning_node(state: UserInputString) -> ReasoningOutput:
        st.write("💡 **Generating Reasoning...**")
        reasoner = ReasoningNode()
        reasoning_output = reasoner.generate_reasoning(state)
        st.write("🧠 **Reasoning Output:**", reasoning_output.dict())
        time.sleep(1)
        return reasoning_output

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
    graph.add_edge("prediction", END)

    # ✅ **Research & Reasoning Endpoints**
    graph.add_edge("researcher", END)
    graph.add_edge("reasoning", END)

    # ✅ **Compile and Run Workflow**
    workflow = graph.compile()

    test = UserInputString(user_input=user_input, next_state="")

    with st.spinner("⏳ Running Workflow..."):
        output = workflow.invoke(test)
    
    # ✅ **Final Output**
    st.success("✅ Workflow Completed!")
     
