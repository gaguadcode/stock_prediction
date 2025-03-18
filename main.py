from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.workflow import create_workflow
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState  # ✅ Unified workflow state model
from langgraph.pregel.io import AddableValuesDict 

# ✅ FastAPI Setup
app = FastAPI(title="Stock & Research AI Workflow API")

# ✅ Enable CORS (Allows Frontend to Call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Logger
logger = get_logger("WorkflowAPI")

# ✅ Load Workflow (Once)
workflow = create_workflow()

# ✅ Define Input Model (MUST MATCH REACT'S PAYLOAD)
class WorkflowRequest(BaseModel):
    user_input: str  # Expecting a string input from React

# ✅ API Endpoint
@app.post("/run-workflow")
def run_workflow(request: WorkflowRequest):
    """
    Accepts user input as JSON and processes it through the AI workflow.
    """
    logger.info("📥 Received user input: %s", request.user_input)

    # ✅ Initialize state (Ensure it's properly structured)
    initial_state = WorkflowState(user_input=request.user_input, next_state="")

    try:
        logger.info("⏳ Running Workflow...")

        # ✅ Ensure the workflow processes and updates the state correctly
        final_state = workflow.invoke(initial_state)

        # 🚨 Debugging: Print the type of `final_state`
        logger.info(f"🛠️ Debug: Workflow returned type: {type(final_state)}")

        # ✅ If workflow returns `AddableValuesDict`, convert it back to `WorkflowState`
        if isinstance(final_state, AddableValuesDict):
            logger.warning("⚠️ Workflow returned AddableValuesDict instead of WorkflowState. Converting...")
            final_state = WorkflowState(**final_state)  # ✅ Convert safely

        # ✅ Ensure correct type is returned
        if not isinstance(final_state, WorkflowState):
            raise TypeError(f"❌ Unexpected return type from workflow: {type(final_state)}")

        # ✅ Convert the final state to a JSON-friendly format
        response = final_state.model_dump()
        logger.info("✅ Workflow Completed Successfully.")

        return {"workflow_output": response}

    except TypeError as e:
        logger.error(f"❌ TypeError: {e} - Likely caused by incorrect state updates.")
        raise HTTPException(status_code=500, detail="Internal Server Error: State Mismatch. Please try again.")

    except Exception as e:
        logger.error(f"❌ Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")