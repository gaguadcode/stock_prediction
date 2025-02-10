# Stock Research and Analysis App

This project provides a streamlined workflow for processing natural language queries about stock prices and predictions. It extracts structured data from user input and generates detailed analysis reports using advanced models.
### User Interface  
![User Interface](/image1.png)
 
![User Interface](/image2.png)
---

## Features

- **Natural Language Processing**: Handles user queries in natural language (e.g., Spanish or English).  
- **Data Extraction**: Extracts structured data from input text using `StockDataExtractor`.  
- **Detailed Analysis**: Performs in-depth stock research and prediction analysis with `StockResearchAgent`.  
- **Interactive Interface**: Provides a user-friendly Streamlit app for seamless interaction.

---

## Installation

1. **Clone this repository**:
   ~~~bash
   git clone https://github.com/your-username/stock-analysis-app.git
   cd stock-analysis-app
   ~~~

2. **Install dependencies**:
   ~~~bash
   pip install -r requirements.txt
   ~~~

3. **Configure .env file**:
   Configure the .env file with the following variables:
   ALPHAVANTAGE_API_KEY = "XXXXX"
   ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
   OUTPUT_CSV = "app/db/stock_data.csv"  

4. **Run the Streamlit app**:
   ~~~bash
   streamlit run app.py
   ~~~

---

## Usage

### Streamlit App

1. **Input Query**  
   Enter your query about stock prices or predictions in the text area.  
   Example:  
   *"quiero saber el precio de IBM de manera mensual para enero de 2025"*

2. **Select Models**  
   - Choose the model for the data extractor (`mistral` or others).
   - Choose the model for the research agent (`mistral` or others).

3. **Analyze**  
   - Click **Analyze Stock** to process your query and view the analysis report.

### Python Script

To use the processing function directly in a Python script:

~~~python
from app import process_and_analyze_stock

input_text = "quiero saber el precio de IBM de manera mensual para enero de 2025"
final_report = process_and_analyze_stock(input_text)
print(final_report)
~~~

---

## Modules Overview

- **StockDataExtractor**  
  Extracts structured data from unstructured natural language input.

- **StockResearchAgent**  
  Conducts research and generates analysis based on extracted data.

---

## Core Workflow

1. **Data Extraction**  
   `StockDataExtractor` processes the input and extracts key information.

2. **Research and Analysis**  
   `StockResearchAgent` generates a comprehensive report.

---

## Example Query

- **Input**:  
  *"quiero saber el precio de IBM de manera mensual para enero de 2025"*

- **Output**:  
  Detailed analysis report of IBM's predicted monthly stock price for January 2025.

---

## Requirements

- Python 3.7+
- Streamlit
- Asyncio-compatible libraries for research and extraction

---

## Contributing

1. **Fork the repository**  
2. **Create a feature branch**:
   ~~~bash
   git checkout -b feature-name
   ~~~
3. **Commit your changes**:
   ~~~bash
   git commit -m 'Add new feature'
   ~~~
4. **Push to the branch**:
   ~~~bash
   git push origin feature-name
   ~~~
5. **Create a Pull Request**

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

- **Gustavo Aguado** - Initial Work  

For questions or suggestions, feel free to contact us or open an issue on GitHub.
