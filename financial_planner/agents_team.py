import asyncio
import datetime
import logging
import time

import requests
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core import CancellationToken
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion,
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.memory.null_memory import NullMemory
from tzlocal import get_localzone

from financial_planner import ANTHROPIC_API_KEY, PERPLEXITY_API_KEY, display_terminal

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_current_date() -> str:
    local_tz = get_localzone()
    now_aware = datetime.datetime.now(local_tz)
    return now_aware.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z (%z)")


def format_enhanced_query(
    query, risk_tolerance=None, time_horizon=None, annual_gross_income=None
):
    current_date = get_current_date()

    enhanced_query = f"Today is {current_date}.\n\n" + query.strip()

    financial_profile = []
    if risk_tolerance:
        financial_profile.append(f"Risk tolerance: {risk_tolerance}")
    if time_horizon:
        financial_profile.append(f"Investment time horizon: {time_horizon}")
    if annual_gross_income is not None:
        financial_profile.append(
            f"Annual gross income: ${float(annual_gross_income):,.2f}"
        )

    if financial_profile:
        profile_text = "\n\n---\n\nMy financial profile:\n" + "\n".join(
            financial_profile
        )
        enhanced_query += profile_text

    enhanced_query += "\n\n---\n\nPlease don't ask any follow up questions. Make reasonable assumptions and provide the best possible answer."

    return enhanced_query


def perplexity_search(query: str, api_key: str, max_retries: int = 3) -> str:
    url = "https://api.perplexity.ai/chat/completions"

    system_instructions = """You are a factual financial search assistant. Answer queries precisely using *only* the provided search context. Assume reasonable details if information is missing. Provide direct, thorough answers formatted for clarity. Do not ask follow-up questions. Today is {current_date}."""

    payload = {
        "model": "sonar-reasoning-pro",
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": query},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 8000,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: HTTP Error: {e}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries}: Request Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
        except ValueError as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries}: JSON Decode Error: {e}")
            return None

    md_output = f"*Search performed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

    choices = data.get("choices", [])
    if not choices:
        logger.warning(
            "No results found in Perplexity API response for query: '%s'. "
            "Consider refining the search terms.",
            query,
        )
        return "No specific information found. Please try refining your query."

    for choice in choices:
        content = choice.get("message", {}).get("content", "")
        md_output += content + "\n\n---\n\n"

    citations = data.get("citations", [])
    if citations:
        md_output += "### Citations / References\n"
        for idx, citation in enumerate(citations, start=1):
            if isinstance(citation, dict):
                url = citation.get("url", "")
                title = citation.get("title", "")
                if url and title:
                    md_output += f"{idx}. [{title}]({url})\n"
                elif url:
                    md_output += f"{idx}. {url}\n"
                else:
                    md_output += f"{idx}. {citation}\n"
            elif isinstance(citation, str):
                md_output += f"{idx}. {citation}\n"
            else:
                logger.warning(f"Unexpected citation type: {type(citation)}")
                md_output += f"{idx}. {citation}\n"

    return md_output


async def create_agent(
    name: str,
    model_client,
    system_message: str,
    description: str,
    tools=None,
    reflect_on_tool_use=False,
    shared_memory: ListMemory = None,
):
    agent_memory = ListMemory(name=f"{name}_memory")

    # Build memory list with agent-specific and shared memory if provided
    memory_list = [agent_memory]
    if shared_memory is not None:
        memory_list.append(shared_memory)

    agent = AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=system_message,
        description=description,
        tools=tools,
        reflect_on_tool_use=reflect_on_tool_use,
        memory=memory_list,
    )

    return agent


async def create_web_search_agent(api_key: str, shared_memory: ListMemory = None):
    def search_tool(query: str) -> str:
        result = perplexity_search(query, api_key)
        if result is None:
            return "Error: Unable to perform the search."
        return result

    current_date = get_current_date()

    system_message = (
        f"You are a financial analyst. Today is {current_date}. Your primary function is to use the web search tool for current financial data. "
        "Provide accurate, direct answers based on search results, **always citing your sources**. "
        "When reporting data, clearly identify any limitations or inconsistencies in the information retrieved, and prioritize the most relevant and recent information. "
        "Make reasonable assumptions for missing details rather than asking questions - no follow-up questions. "
        "Consult shared memory for relevant client profile information to tailor responses."
    )

    description = "Web Search Agent: Retrieves current financial info via web search, providing cited, direct answers. Use for up-to-date market data, regulations, or news."

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o", timeout=60, temperature=0.0
    )

    web_search_agent = await create_agent(
        name="web_search_agent",
        model_client=model_client,
        system_message=system_message,
        description=description,
        tools=[search_tool],
        reflect_on_tool_use=True,
        shared_memory=shared_memory,
    )

    return web_search_agent


async def create_code_executor_agent(work_dir: str = "coding", timeout: int = 30):
    try:
        code_executor = DockerCommandLineCodeExecutor(
            image="jupyter/scipy-notebook",
            timeout=timeout,
            work_dir=work_dir,
            init_command="pip install --quiet seaborn scikit-learn",
            auto_remove=True,
        )

        await code_executor.start()
        await asyncio.sleep(2)

        code_executor_agent = CodeExecutorAgent(
            name="code_executor_agent",
            code_executor=code_executor,
            description="Code Executor Agent: Executes Python code snippets provided in markdown blocks (```python). Must use this *after* the Code Writer Agent generates code to get the output/results.",
        )
        return code_executor_agent, code_executor
    except Exception as e:
        logger.exception("Error creating code executor agent: %s", e)
        raise


async def create_code_writer_agent(shared_memory: ListMemory = None):
    current_date = get_current_date()

    system_message = (
        f"You are a financial code writer. Today is {current_date}. Generate clear, commented Python code to solve financial problems. "
        "**IMPORTANT CONSTRAINT: You MUST only use standard Python libraries OR libraries included in the 'jupyter/scipy-notebook' Docker image environment.** "
        "Key available libraries include, but are not limited to: **Pandas, NumPy, SciPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn, SQLAlchemy, Openpyxl, Beautifulsoup4, Bokeh, Dask, h5py, Numba, Patsy, Sympy, PyTables, Scikit-image**. "
        "Do NOT import or use any libraries outside of standard Python and this specific pre-installed set. "
        "Ensure code is in proper markdown blocks (```python). Include print statements for all results. "
        "Review previous messages for available data before coding. Document key assumptions in comments. When generating financial models, output all variables and weights that inform the final recommendation. "
        "Make reasonable assumptions for missing details. Consult shared memory for client profile context if relevant for the calculation."
        "Use the code executor agent immediately after this to run the generated code and return the results."
    )

    description = "Code Writer Agent: Writes commented Python code (using only standard Python or libraries available in jupyter/scipy-notebook like Pandas, NumPy, SciPy, etc.) in markdown for financial calculations. Any code generated should be run next by the code executor agent to get the output/results."

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o", timeout=60, temperature=0.0
    )

    code_writer_agent = await create_agent(
        name="code_writer_agent",
        model_client=model_client,
        system_message=system_message,
        description=description,
        shared_memory=shared_memory,
    )

    return code_writer_agent


async def create_financial_advisor_agent(shared_memory: ListMemory = None):
    current_date = get_current_date()

    system_message = (
        f"You are a professional financial advisor. Today is {current_date}. Carefully review all the messages and synthesize information (from user, search, code execution if available) to provide clear, **actionable recommendations**. "
        "**Explain your reasoning, potential risks, and limitations clearly.** Base your advice on available data, making reasonable assumptions for missing details. Do not ask follow-up questions. "
        "Consult shared memory for the client's financial profile to personalize advice."
    )

    description = "Financial Advisor Agent: Analyzes all available information to provide synthesized, actionable financial advice, explaining risks and reasoning. Considers client profile."

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o", timeout=60, temperature=0.0
    )

    financial_advisor_agent = await create_agent(
        name="financial_advisor_agent",
        model_client=model_client,
        system_message=system_message,
        description=description,
        shared_memory=shared_memory,
    )

    return financial_advisor_agent


async def create_financial_team(
    perplexity_api_key: str,
    anthropic_api_key: str,
    risk_tolerance: str = None,
    time_horizon: str = None,
    annual_gross_income: float = None,
):
    shared_memory = ListMemory(name="financial_profile")

    profile_items = []
    if risk_tolerance:
        profile_items.append(f"Risk tolerance: {risk_tolerance}")
    if time_horizon:
        profile_items.append(f"Investment time horizon: {time_horizon}")
    if annual_gross_income:
        profile_items.append(f"Annual gross income: ${annual_gross_income:,.2f}")

    if profile_items:
        financial_profile = "Client financial profile:\n" + "\n".join(profile_items)
        await shared_memory.add(
            MemoryContent(content=financial_profile, mime_type="text/plain")
        )

    web_search_agent = await create_web_search_agent(
        perplexity_api_key, shared_memory=shared_memory
    )
    code_writer_agent = await create_code_writer_agent(shared_memory=shared_memory)
    code_executor_agent, code_executor = await create_code_executor_agent()
    financial_advisor_agent = await create_financial_advisor_agent(
        shared_memory=shared_memory
    )

    anthropic_orchestrator = AnthropicChatCompletion(
        ai_model_id="claude-3-5-sonnet-20241022", api_key=anthropic_api_key
    )
    orchestrator_settings = AnthropicChatPromptExecutionSettings(
        temperature=0.0, max_tokens=4096
    )
    sk_kernel_orchestrator = Kernel(memory=NullMemory())
    claude_orchestrator_client = SKChatCompletionAdapter(
        anthropic_orchestrator,
        kernel=sk_kernel_orchestrator,
        prompt_settings=orchestrator_settings,
    )

    termination_condition = MaxMessageTermination(max_messages=30)

    team = MagenticOneGroupChat(
        participants=[
            web_search_agent,
            code_writer_agent,
            code_executor_agent,
            financial_advisor_agent,
        ],
        termination_condition=termination_condition,
        model_client=claude_orchestrator_client,
    )

    return team, code_executor


async def test_web_search_agent():
    """
    Test function to verify the web search agent's functionality.
    """
    try:
        shared_memory = ListMemory(name="financial_profile")
        await shared_memory.add(
            MemoryContent(
                content="Risk tolerance: moderate\nInvestment time horizon: 10 years",
                mime_type="text/plain",
            )
        )

        agent = await create_web_search_agent(
            PERPLEXITY_API_KEY, shared_memory=shared_memory
        )

        test_message = TextMessage(
            content="What are the current best practices for retirement savings based on my risk tolerance and time horizon?",
            source="user",
        )

        token = CancellationToken()

        print("Sending query to agent...")
        response = await agent.on_messages([test_message], token)

        print(f"Agent Response:\n{response.chat_message.content}")
        return response

    except Exception as e:
        logger.exception("Error during test: %s", e)
        raise


async def test_code_executor_agent():
    """
    Test function to verify the code executor agent's functionality.
    """
    code_executor_agent = None

    try:
        code_executor_agent, code_executor = await create_code_executor_agent()

        test_message = TextMessage(
            content='''Here's a Python script to calculate compound interest:
```python
def calculate_compound_interest(principal, rate, time, compounds_per_year=12):
    """Calculate compound interest."""
    amount = principal * (1 + rate/compounds_per_year)**(compounds_per_year*time)
    interest = amount - principal
    return amount, interest

# Example calculation
principal = 10000
rate = 0.05  # 5% annual interest
time = 10    # 10 years

final_amount, earned_interest = calculate_compound_interest(principal, rate, time)
print(f"Initial Investment: ${principal:,.2f}")
print(f"After {time} years at {rate*100}% annual interest:")
print(f"Final Amount: ${final_amount:,.2f}")
print(f"Interest Earned: ${earned_interest:,.2f}")
```''',
            source="user",
        )

        token = CancellationToken()

        print("Sending code to Code Executor Agent...")
        response = await code_executor_agent.on_messages([test_message], token)

        print(f"Code Execution Result:\n{response.chat_message.content}")
        return response

    except Exception as e:
        logger.exception("Error during code executor test: %s", e)
        raise
    finally:
        if code_executor:
            try:
                await code_executor.stop()
            except Exception as e:
                logger.warning(f"Error stopping code executor: {e}")


async def test_code_writer_agent():
    """
    Test function to verify the code writer agent's functionality.
    """
    try:
        shared_memory = ListMemory(name="financial_profile")
        await shared_memory.add(
            MemoryContent(
                content="Risk tolerance: moderate\nAnnual gross income: $120,000.00",
                mime_type="text/plain",
            )
        )

        agent = await create_code_writer_agent(shared_memory=shared_memory)

        test_message = TextMessage(
            content=(
                "I have $5000 to invest at a simple annual interest rate of 3.5% for 5 years. "
                "Write Python code to calculate the final value of the investment."
            ),
            source="user",
        )

        token = CancellationToken()

        print("Sending query to Code Writer Agent...")
        response = await agent.on_messages([test_message], token)

        print(f"Agent Response (Generated Code):\n{response.chat_message.content}")
        return response

    except Exception as e:
        logger.exception("Error during code writer test: %s", e)
        raise


async def test_financial_advisor_agent():
    try:
        shared_memory = ListMemory(name="financial_profile")
        await shared_memory.add(
            MemoryContent(
                content="Risk tolerance: moderate\nInvestment time horizon: long-term\nAnnual gross income: $95,000.00",
                mime_type="text/plain",
            )
        )

        agent = await create_financial_advisor_agent(shared_memory=shared_memory)

        test_message = TextMessage(
            content=(
                "I have $50,000 saved and I'm a mid-career professional. "
                "Considering current market conditions, what diversified investment strategies would you recommend for growth?"
            ),
            source="user",
        )

        token = CancellationToken()

        print("Sending query to the financial advisor agent...")
        response = await agent.on_messages([test_message], token)

        print("Financial Advisor Agent Response:\n", response.chat_message.content)
        return response

    except Exception as e:
        logger.exception("Error during financial advisor agent test: %s", e)
        raise


async def test_financial_team():
    try:
        risk_tolerance = "Moderate - Balanced Growth"
        time_horizon = "Long Term (5+ years)"
        annual_gross_income = None

        team, code_executor = await create_financial_team(
            perplexity_api_key=PERPLEXITY_API_KEY,
            anthropic_api_key=ANTHROPIC_API_KEY,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon,
            annual_gross_income=annual_gross_income,
        )

        user_query = """I have $5000 to invest at a simple annual interest rate of 3.5% for 5 years. Write Python code to calculate the final value of the investment."""

        enhanced_content = format_enhanced_query(
            user_query, risk_tolerance, time_horizon, annual_gross_income
        )
        enhanced_message = TextMessage(content=enhanced_content, source="user")

        token = CancellationToken()
        print("Sending query to financial team...")

        async for event in team.run_stream(
            task=[enhanced_message], cancellation_token=token
        ):
            display_terminal.pretty_print_event(event)

    except Exception as e:
        logger.exception("Error during financial team test: %s", e)
        raise
    finally:
        if code_executor:
            try:
                await code_executor.stop()
            except Exception as e:
                logger.warning(f"Error stopping code executor: {e}")


async def run_all_tests():
    try:
        # Test perplexity_search function
        logger.info("Testing perplexity_search function...")
        query = "What are the current best practices for retirement savings?"
        result = perplexity_search(query, PERPLEXITY_API_KEY)
        print(result)

        print("\n---\n")

        # Test Web Search Agent
        logger.info("Testing Web Search Agent...")
        await test_web_search_agent()

        print("\n---\n")

        # Test Code Executor Agent
        logger.info("Testing Code Executor Agent...")
        await test_code_executor_agent()

        print("\n---\n")

        # Test Code Writer Agent
        logger.info("Testing Code Writer Agent...")
        await test_code_writer_agent()

        print("\n---\n")

        # Test Financial Advisor Agent
        logger.info("Testing Financial Advisor Agent...")
        await test_financial_advisor_agent()

        print("\n---\n")

        # Test Financial Team
        logger.info("Testing Financial Team...")
        await test_financial_team()

    except Exception as e:
        logger.exception("Error during tests: %s", e)
        raise


if __name__ == "__main__":
    logger.info("Starting Agent Tests...")
    asyncio.run(run_all_tests())
