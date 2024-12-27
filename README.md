Stock Research and Analysis App

This project provides a streamlined workflow for processing natural language queries about stock prices and predictions. It extracts structured data from user input and generates detailed analysis reports using advanced models.

Features

Natural Language Processing: Handles user queries in natural language (e.g., Spanish or English).
Data Extraction: Extracts structured data from input text using StockDataExtractor.
Detailed Analysis: Performs in-depth stock research and prediction analysis with StockResearchAgent.
Interactive Interface: Provides a user-friendly Streamlit app for seamless interaction.
Installation

Clone this repository:
git clone https://github.com/your-username/stock-analysis-app.git
cd stock-analysis-app
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
Usage

Streamlit App
Input Query: Enter your query about stock prices or predictions in the text area (e.g., "quiero saber el precio de IBM de manera mensual para enero de 2025").
Select Models:
Choose the model for the data extractor (mistral or others).
Choose the model for the research agent (mistral or others).
Analyze: Click "Analyze Stock" to process your query and view the analysis report.
Python Script
To use the processing function directly in a Python script:

from app import process_and_analyze_stock

input_text = "quiero saber el precio de IBM de manera mensual para enero de 2025"
final_report = process_and_analyze_stock(input_text)
print(final_report)
Modules Overview

StockDataExtractor: Extracts structured data from unstructured natural language input.
StockResearchAgent: Conducts research and generates analysis based on extracted data.
Core Workflow
Data Extraction: StockDataExtractor processes input and extracts key information.
Research and Analysis: StockResearchAgent generates a comprehensive report.
Example Query

Input: "quiero saber el precio de IBM de manera mensual para enero de 2025"
Output: Detailed analysis report of IBM's predicted monthly stock price for January 2025.
Requirements

Python 3.7+
Streamlit
Asyncio-compatible libraries for research and extraction
Contributing

Fork the repository.
Create a feature branch: git checkout -b feature-name.
Commit your changes: git commit -m 'Add new feature'.
Push to the branch: git push origin feature-name.
Create a Pull Request.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Authors

Your Name - Initial Work
For questions or suggestions, feel free to contact us or open an issue on GitHub.