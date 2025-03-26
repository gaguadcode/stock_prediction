import sys
import os
from app.utils.logger import get_logger
from app.utils.datatypes import WorkflowState
from app import workflow  # Replace `your_module_path` with the actual module name

logger = get_logger("WorkflowTest")

def read_markdown_file(file_path):
    """Reads content from a Markdown file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def main(file_path):
    """Executes the workflow using input from a Markdown file."""
    user_input = read_markdown_file(file_path)

    # Initialize workflow
    workflow = workflow()

    # Create an initial state
    initial_state = WorkflowState(user_input=user_input, next_state="router")

    # Execute the workflow
    logger.info("🚀 Running Workflow...")
    final_state = workflow.invoke(initial_state)

    # Log & print results
    logger.info("✅ Workflow Execution Complete!")
    logger.info("🔍 Final State Output: %s", final_state.model_dump())

    print("\n=== Workflow Execution Complete ===")
    print(final_state.model_dump())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_workflow.py <path_to_markdown_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)
