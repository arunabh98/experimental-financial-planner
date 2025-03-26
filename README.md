---
title: Financial Planner
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: true
license: agpl-3.0
short_description: AI-driven personal financial planning assistant
tags:
  - finance
  - planning
  - ai
  - ml
  - llm
  - agents
startup_duration_timeout: "10m"
---

# Financial Planner

**Financial Planner** is an AI-driven personal financial planning assistant designed to provide accurate, well-researched financial insights and recommendations. Leveraging a multi-agent architecture, it integrates web search, Python code generation, and secure code execution to help users make informed financial decisions.

## Features

- **Multi-Agent Architecture:**  
  Combines specialized agents:
  - **Web Search Agent:** Retrieves up-to-date financial information via the Perplexity API.
  - **Code Writer Agent:** Generates Python code for financial calculations.
  - **Code Executor Agent:** Executes Python code safely in a Docker container.
  - **Financial Advisor Agent:** Analyzes data and provides personalized financial recommendations.
  
- **Interactive Web Interface:**  
  A basic interface built with FastAPI that displays the steps the agents are taking along with the final answer to the user's question.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/arunabh98/financial-planner
   cd financial-planner
   ```

2. **Install Dependencies with Poetry:**

   This project uses [Poetry](https://python-poetry.org/) for dependency management.

   ```bash
   poetry install
   ```

3. **Configure Environment Variables:**

   Create a `.env` file (which is excluded from version control) and add your API keys:

   ```env
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Start the Application:**

   Launch the FastAPI server using Uvicorn:

   ```bash
   poetry run uvicorn financial_planner.app:app --reload
   ```

2. **Access the Web Interface:**

   Open your browser and navigate to [http://localhost:8000](http://localhost:8000) to start asking your financial questions.

## License

This project is licensed under the **GNU Affero General Public License v3**.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions or bug fixes.

## Author

**Arunabh Ghosh**
