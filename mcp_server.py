#!/usr/bin/env python3
"""
MCP Server for Stock Prediction and Analysis

This MCP server exposes the stock prediction pipeline, research, and reasoning
capabilities as documented tools that LLMs can consume via the Model Context Protocol.

Tools available:
- route_user_input: Routes user input to appropriate pipeline (stock, research, reasoning)
- extract_stock_entities: Extracts stock symbol, period, and target dates from natural language
- fetch_stock_data: Fetches historical stock data from Alpha Vantage API
- train_stock_model: Trains ML model on historical data
- predict_stock_price: Generates stock price predictions
- research_topic: Researches topics using Wikipedia
- generate_reasoning: Generates analytical reasoning

Run with: python mcp_server.py
Or via MCP: mcp run mcp_server.py
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

# Import existing nodes
from app.graph_nodes.main_router import MainRouterNode
from app.graph_nodes.stock.stock_extractor_agent import StockDataExtractor
from app.graph_nodes.stock.stock_fetch_data import HistoricalDataFetcher
from app.graph_nodes.stock.model_training import StockDataTrainer
from app.graph_nodes.stock.prediction import StockPredictor
from app.graph_nodes.research.web_researcher_agent import ResearcherNode
from app.graph_nodes.reasoning.reasoning_agent import ReasoningNode

# Import data types
from app.utils.datatypes import (
    UserInputString,
    EntityExtractOutput,
    DataFetchOutput,
    FinalPredictionState,
    ResearchOutput,
)
from app.utils.logger import get_logger

logger = get_logger("MCPServer")

# Initialize the MCP server
server = Server("stock-prediction-mcp")

# Initialize node instances (lazy loading for efficiency)
_router_node = None
_extractor_node = None
_fetcher_node = None
_trainer_node = None
_researcher_node = None
_reasoning_node = None


def get_router_node() -> MainRouterNode:
    global _router_node
    if _router_node is None:
        _router_node = MainRouterNode()
    return _router_node


def get_extractor_node() -> StockDataExtractor:
    global _extractor_node
    if _extractor_node is None:
        _extractor_node = StockDataExtractor()
    return _extractor_node


def get_fetcher_node() -> HistoricalDataFetcher:
    global _fetcher_node
    if _fetcher_node is None:
        _fetcher_node = HistoricalDataFetcher()
    return _fetcher_node


def get_trainer_node() -> StockDataTrainer:
    global _trainer_node
    if _trainer_node is None:
        _trainer_node = StockDataTrainer()
    return _trainer_node


def get_researcher_node() -> ResearcherNode:
    global _researcher_node
    if _researcher_node is None:
        _researcher_node = ResearcherNode()
    return _researcher_node


def get_reasoning_node() -> ReasoningNode:
    global _reasoning_node
    if _reasoning_node is None:
        _reasoning_node = ReasoningNode()
    return _reasoning_node


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools in the MCP server.
    """
    return [
        Tool(
            name="route_user_input",
            description="""Routes user input to the appropriate processing pipeline.

Uses a hybrid approach combining keyword matching and LLM-based analysis to determine
whether the input should be processed by:
- entity_extraction: For stock price queries, financial markets, investing
- researcher: For general knowledge, Wikipedia topics, explanations
- reasoning: For analysis, critical thinking, predictions

Returns the recommended route along with the original user input.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The natural language input from the user to be routed"
                    }
                },
                "required": ["user_input"]
            }
        ),
        Tool(
            name="extract_stock_entities",
            description="""Extracts structured stock information from natural language input.

Uses an LLM to parse the user's query and extract:
- stock_symbol: The stock ticker (e.g., "IBM", "AAPL", "GOOGL")
- date_period: Time series granularity (TIME_SERIES_MONTHLY, TIME_SERIES_WEEKLY, TIME_SERIES_DAILY)
- date_target: List of target prediction dates in YYYY-MM-DD format

Example input: "I want to know IBM stock prices monthly for January 2025"
Example output: {"stock_symbol": "IBM", "date_period": "TIME_SERIES_MONTHLY", "date_target": ["2025-01-01"]}""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Natural language query about stock prediction"
                    }
                },
                "required": ["user_input"]
            }
        ),
        Tool(
            name="fetch_stock_data",
            description="""Fetches historical stock data from Alpha Vantage API and stores it in PostgreSQL.

Takes the extracted entity information and:
1. Creates a PostgreSQL database if it doesn't exist (named {symbol}_{period})
2. Fetches historical price data from Alpha Vantage API
3. Processes and stores the data in the database

Returns the database URL for subsequent training operations.

Requires: stock_symbol, date_period, date_target (from extract_stock_entities output)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Original user input"
                    },
                    "stock_symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'IBM', 'AAPL')"
                    },
                    "date_period": {
                        "type": "string",
                        "enum": ["TIME_SERIES_MONTHLY", "TIME_SERIES_WEEKLY", "TIME_SERIES_DAILY"],
                        "description": "Time series granularity"
                    },
                    "date_target": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target dates in YYYY-MM-DD format"
                    }
                },
                "required": ["user_input", "stock_symbol", "date_period", "date_target"]
            }
        ),
        Tool(
            name="train_stock_model",
            description="""Trains a Gradient Boosting model on historical stock data.

The training process:
1. Fetches data from the PostgreSQL database
2. Transforms dates into ML-friendly features (cyclical encoding)
3. Splits data (80% train, 20% test)
4. Trains a GradientBoostingRegressor model
5. Evaluates with Mean Squared Error (MSE)
6. Stores the trained model in Redis

Returns the MSE score and metadata for prediction.

Requires: Output from fetch_stock_data (database_url and entity information)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Original user input"
                    },
                    "stock_symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "date_period": {
                        "type": "string",
                        "enum": ["TIME_SERIES_MONTHLY", "TIME_SERIES_WEEKLY", "TIME_SERIES_DAILY"],
                        "description": "Time series granularity"
                    },
                    "date_target": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target dates in YYYY-MM-DD format"
                    },
                    "database_url": {
                        "type": "string",
                        "description": "PostgreSQL database URL from fetch_stock_data"
                    }
                },
                "required": ["user_input", "stock_symbol", "date_period", "date_target", "database_url"]
            }
        ),
        Tool(
            name="predict_stock_price",
            description="""Generates stock price predictions using a trained model.

Loads the trained Gradient Boosting model from Redis and predicts prices for the target dates.
Uses the same feature transformation (cyclical date encoding) as training.

Returns:
- predictions: List of predicted stock prices
- mse: Model's mean squared error (accuracy indicator)
- metadata: Stock symbol, dates, database URL

Requires: Output from train_stock_model (trained model in Redis, MSE, metadata)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Original user input"
                    },
                    "stock_symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "date_period": {
                        "type": "string",
                        "enum": ["TIME_SERIES_MONTHLY", "TIME_SERIES_WEEKLY", "TIME_SERIES_DAILY"],
                        "description": "Time series granularity"
                    },
                    "date_target": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target dates in YYYY-MM-DD format"
                    },
                    "database_url": {
                        "type": "string",
                        "description": "PostgreSQL database URL"
                    },
                    "mse": {
                        "type": "number",
                        "description": "Mean Squared Error from training"
                    }
                },
                "required": ["user_input", "stock_symbol", "date_period", "date_target", "database_url", "mse"]
            }
        ),
        Tool(
            name="research_topic",
            description="""Researches a topic using Wikipedia.

Process:
1. Uses LLM to extract the most relevant Wikipedia topic from the query
2. Fetches the Wikipedia page summary for that topic
3. Returns the research findings

Useful for general knowledge questions, definitions, historical information, etc.

Example: "What is the history of IBM?" -> Returns IBM Wikipedia summary""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Natural language query for research"
                    }
                },
                "required": ["user_input"]
            }
        ),
        Tool(
            name="generate_reasoning",
            description="""Generates analytical reasoning using an LLM.

Takes user input or research data and produces:
- Critical analysis of the information
- Implications and significance
- Logical reasoning about the topic

Can be used:
1. Directly with user input for analysis questions
2. After research_topic to analyze Wikipedia findings
3. For any query requiring analytical thinking

Input can include research_output from prior research_topic call for enhanced reasoning.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The user's query or statement to analyze"
                    },
                    "research_output": {
                        "type": "string",
                        "description": "Optional: Research findings from research_topic to include in analysis"
                    }
                },
                "required": ["user_input"]
            }
        ),
        Tool(
            name="run_full_stock_pipeline",
            description="""Runs the complete stock prediction pipeline in one call.

This is a convenience tool that executes all stock prediction steps:
1. Extract entities (stock symbol, period, dates)
2. Fetch historical data from Alpha Vantage
3. Train Gradient Boosting model
4. Generate predictions

Returns comprehensive results including predictions, accuracy (MSE), and all metadata.

Use this for simple stock prediction queries instead of calling individual tools.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "Natural language stock prediction query"
                    }
                },
                "required": ["user_input"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle tool calls from the MCP client.
    """
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "route_user_input":
            return await handle_route_user_input(arguments)
        elif name == "extract_stock_entities":
            return await handle_extract_stock_entities(arguments)
        elif name == "fetch_stock_data":
            return await handle_fetch_stock_data(arguments)
        elif name == "train_stock_model":
            return await handle_train_stock_model(arguments)
        elif name == "predict_stock_price":
            return await handle_predict_stock_price(arguments)
        elif name == "research_topic":
            return await handle_research_topic(arguments)
        elif name == "generate_reasoning":
            return await handle_generate_reasoning(arguments)
        elif name == "run_full_stock_pipeline":
            return await handle_full_stock_pipeline(arguments)
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "tool": name})
        )]


async def handle_route_user_input(arguments: dict) -> list[TextContent]:
    """Route user input to appropriate pipeline."""
    user_input = arguments.get("user_input")
    if not user_input:
        return [TextContent(type="text", text=json.dumps({"error": "user_input is required"}))]

    router = get_router_node()
    input_state = UserInputString(user_input=user_input, next_state="")

    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, router.determine_route, input_state)

    return [TextContent(
        type="text",
        text=json.dumps({
            "route": result.next_state,
            "user_input": result.user_input,
            "description": {
                "entity_extraction": "Route to stock prediction pipeline",
                "researcher": "Route to Wikipedia research",
                "reasoning": "Route to analytical reasoning"
            }.get(result.next_state, "Unknown route")
        }, indent=2)
    )]


async def handle_extract_stock_entities(arguments: dict) -> list[TextContent]:
    """Extract stock entities from natural language."""
    user_input = arguments.get("user_input")
    if not user_input:
        return [TextContent(type="text", text=json.dumps({"error": "user_input is required"}))]

    extractor = get_extractor_node()
    input_state = UserInputString(user_input=user_input, next_state="entity_extraction")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, extractor.process_input, input_state)

    return [TextContent(
        type="text",
        text=json.dumps({
            "user_input": result.user_input,
            "stock_symbol": result.stock_symbol,
            "date_period": result.date_period,
            "date_target": result.date_target
        }, indent=2)
    )]


async def handle_fetch_stock_data(arguments: dict) -> list[TextContent]:
    """Fetch historical stock data."""
    required = ["user_input", "stock_symbol", "date_period", "date_target"]
    for field in required:
        if field not in arguments:
            return [TextContent(type="text", text=json.dumps({"error": f"{field} is required"}))]

    fetcher = get_fetcher_node()
    entity_output = EntityExtractOutput(
        user_input=arguments["user_input"],
        stock_symbol=arguments["stock_symbol"],
        date_period=arguments["date_period"],
        date_target=arguments["date_target"]
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fetcher.fetch_and_store_historical_data, entity_output)

    return [TextContent(
        type="text",
        text=json.dumps({
            "user_input": result.user_input,
            "stock_symbol": result.stock_symbol,
            "date_period": result.date_period,
            "date_target": result.date_target,
            "database_url": result.database_url,
            "status": "Data fetched and stored successfully"
        }, indent=2)
    )]


async def handle_train_stock_model(arguments: dict) -> list[TextContent]:
    """Train ML model on historical data."""
    required = ["user_input", "stock_symbol", "date_period", "date_target", "database_url"]
    for field in required:
        if field not in arguments:
            return [TextContent(type="text", text=json.dumps({"error": f"{field} is required"}))]

    trainer = get_trainer_node()
    data_fetch_output = DataFetchOutput(
        user_input=arguments["user_input"],
        stock_symbol=arguments["stock_symbol"],
        date_period=arguments["date_period"],
        date_target=arguments["date_target"],
        database_url=arguments["database_url"]
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, trainer.execute_training, data_fetch_output)

    return [TextContent(
        type="text",
        text=json.dumps({
            "user_input": result.user_input,
            "stock_symbol": result.stock_symbol,
            "date_period": result.date_period,
            "date_target": result.date_target,
            "database_url": result.database_url,
            "mse": result.mse,
            "status": "Model trained and saved to Redis"
        }, indent=2)
    )]


async def handle_predict_stock_price(arguments: dict) -> list[TextContent]:
    """Generate stock price predictions."""
    required = ["user_input", "stock_symbol", "date_period", "date_target", "database_url", "mse"]
    for field in required:
        if field not in arguments:
            return [TextContent(type="text", text=json.dumps({"error": f"{field} is required"}))]

    final_state = FinalPredictionState(
        user_input=arguments["user_input"],
        stock_symbol=arguments["stock_symbol"],
        date_period=arguments["date_period"],
        date_target=arguments["date_target"],
        database_url=arguments["database_url"],
        mse=arguments["mse"]
    )

    predictor = StockPredictor(final_state)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, predictor.make_predictions, final_state)

    return [TextContent(
        type="text",
        text=json.dumps({
            "user_input": result.user_input,
            "stock_symbol": result.stock_symbol,
            "date_period": result.date_period,
            "date_target": result.date_target,
            "predictions": result.predictions,
            "mse": result.mse,
            "prediction_details": [
                {"date": date, "predicted_price": price}
                for date, price in zip(result.date_target, result.predictions)
            ]
        }, indent=2)
    )]


async def handle_research_topic(arguments: dict) -> list[TextContent]:
    """Research a topic using Wikipedia."""
    user_input = arguments.get("user_input")
    if not user_input:
        return [TextContent(type="text", text=json.dumps({"error": "user_input is required"}))]

    researcher = get_researcher_node()
    input_state = UserInputString(user_input=user_input, next_state="researcher")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, researcher.researcher, input_state)

    return [TextContent(
        type="text",
        text=json.dumps({
            "user_input": result.user_input,
            "research_output": result.research_output
        }, indent=2)
    )]


async def handle_generate_reasoning(arguments: dict) -> list[TextContent]:
    """Generate analytical reasoning."""
    user_input = arguments.get("user_input")
    if not user_input:
        return [TextContent(type="text", text=json.dumps({"error": "user_input is required"}))]

    reasoning = get_reasoning_node()

    # Check if research_output is provided (coming from research_topic)
    research_output = arguments.get("research_output")

    if research_output:
        input_state = ResearchOutput(
            user_input=user_input,
            next_state="reasoning",
            research_output=research_output
        )
    else:
        input_state = UserInputString(user_input=user_input, next_state="reasoning")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, reasoning.generate_reasoning, input_state)

    return [TextContent(
        type="text",
        text=json.dumps({
            "user_input": result.user_input,
            "reasoning_output": result.reasoning_output
        }, indent=2)
    )]


async def handle_full_stock_pipeline(arguments: dict) -> list[TextContent]:
    """Run the complete stock prediction pipeline."""
    user_input = arguments.get("user_input")
    if not user_input:
        return [TextContent(type="text", text=json.dumps({"error": "user_input is required"}))]

    try:
        loop = asyncio.get_event_loop()

        # Step 1: Extract entities
        extractor = get_extractor_node()
        input_state = UserInputString(user_input=user_input, next_state="entity_extraction")
        entity_result = await loop.run_in_executor(None, extractor.process_input, input_state)

        # Step 2: Fetch data
        fetcher = get_fetcher_node()
        data_result = await loop.run_in_executor(None, fetcher.fetch_and_store_historical_data, entity_result)

        # Step 3: Train model
        trainer = get_trainer_node()
        train_result = await loop.run_in_executor(None, trainer.execute_training, data_result)

        # Step 4: Make predictions
        predictor = StockPredictor(train_result)
        prediction_result = await loop.run_in_executor(None, predictor.make_predictions, train_result)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "user_input": prediction_result.user_input,
                "stock_symbol": prediction_result.stock_symbol,
                "date_period": prediction_result.date_period,
                "date_target": prediction_result.date_target,
                "predictions": prediction_result.predictions,
                "mse": prediction_result.mse,
                "database_url": prediction_result.database_url,
                "prediction_details": [
                    {"date": date, "predicted_price": round(price, 2)}
                    for date, price in zip(prediction_result.date_target, prediction_result.predictions)
                ]
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Error in full stock pipeline: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "error": str(e),
                "stage": "pipeline_execution"
            }, indent=2)
        )]


async def main():
    """Run the MCP server."""
    logger.info("Starting Stock Prediction MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
