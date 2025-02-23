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
from autogen_core.memory import ListMemory
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion,
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.memory.null_memory import NullMemory

from financial_planner import ANTHROPIC_API_KEY, PERPLEXITY_API_KEY, display_terminal

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def perplexity_search(query: str, api_key: str, max_retries: int = 3) -> str:
    url = "https://api.perplexity.ai/chat/completions"

    system_instructions = """You are a knowledgeable financial assistant. Your job is to answer user queries with factual, precise, and well-researched information relating to financial planning, investments, economic indicators, or general money matters. If the user's query is not strictly financial, you can still provide logical, concise, and accurate details. Always adhere to any requested format. Your answers should be thorough yet easy to understand, enabling users to make informed decisions."""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
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


async def create_web_search_agent(api_key: str):

    def search_tool(query: str) -> str:
        return perplexity_search(query, api_key)

    def get_current_date() -> str:
        return datetime.datetime.now().strftime("%B %d, %Y")

    current_date = get_current_date()
    web_search_memory = ListMemory(name="web_search_history")

    web_search_agent = AssistantAgent(
        name="web_search_agent",
        model_client=OpenAIChatCompletionClient(model="gpt-4o", timeout=60),
        tools=[search_tool],
        system_message=(
            f"You are a highly knowledgeable financial analyst. Today is {current_date}. "
            "Your primary responsibility is to provide accurate, concise, and well-researched answers to user queries "
            "related to financial planning, investments, and economic indicators. "
            "You have access to a powerful web search tool to retrieve the latest information."
        ),
        description="Web Search Agent: Use this agent to retrieve financial information from the web. Provide a query, and it will use the Perplexity API to search and return results in markdown format.",
        reflect_on_tool_use=True,
        memory=[web_search_memory],
    )

    return web_search_agent


async def test_web_search_agent():
    """
    Test function to verify the web search agent's functionality.
    """
    try:
        agent = await create_web_search_agent(PERPLEXITY_API_KEY)

        test_message = TextMessage(
            content="What are the current best practices for retirement savings?",
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


async def create_code_executor_agent(work_dir: str = "coding", timeout: int = 30):
    try:
        code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir, timeout=timeout)

        code_executor_agent = CodeExecutorAgent(
            name="code_executor_agent",
            code_executor=code_executor,
            description="Code Executor Agent: This agent executes Python code snippets. Provide Python code within ```python blocks and it will run the code in a Docker container and return the output.",
        )
        return code_executor_agent
    except Exception as e:
        logger.exception("Error creating code executor agent: %s", e)
        raise


async def test_code_executor_agent():
    """
    Test function to verify the code executor agent's functionality.
    """
    code_executor_agent = None

    try:
        code_executor_agent = await create_code_executor_agent()

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


async def create_code_writer_agent():
    model_client = OpenAIChatCompletionClient(model="gpt-4o", timeout=60)
    code_writer_memory = ListMemory(name="code_writer_memory")

    code_writer_system_message = (
        "You are a skilled financial code writer specializing in Python.  "
        "Your task is to generate Python code to perform financial calculations based on user requests. "
        "Follow these steps:\n"
        "1. Analyze the requirements and briefly state your approach.\n"
        "2. Generate complete, well-documented Python code using only standard libraries. No complex libraries like numpy. STANDARD LIBRARIES.\n"
        "3. Include error handling for edge cases (e.g., invalid input, division by zero).\n"
        "4. Add clear comments explaining any complex calculations or logic.\n"
        "5. Print results in a clear, formatted way (e.g., using f-strings with appropriate formatting).\n"
        "6. Validate user inputs where appropriate (e.g., check for non-negative numbers, valid dates).\n"
        "Enclose all code within Markdown code blocks (`python ... `)."
    )

    code_writer_agent = AssistantAgent(
        name="code_writer_agent",
        model_client=model_client,
        system_message=code_writer_system_message,
        description="Code Writer Agent: This agent writes Python code (using only standard libraries) to perform financial calculations. Ask it to write code (without third-party libraries like numpy) for a specific calculation, and it will generate a Python code snippet within a markdown block.",
        memory=[code_writer_memory],
    )

    return code_writer_agent


async def test_code_writer_agent():
    """
    Test function to verify the code writer agent's functionality.
    """
    try:
        agent = await create_code_writer_agent()

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


async def create_financial_advisor_agent():
    financial_advisor_system_message = """You are a professional financial advisor specializing in providing comprehensive financial guidance. Your role is to:

1. Analyze financial information, market conditions, investment options, and economic trends
2. Synthesize complex financial data into clear, actionable insights
3. Consider each situation's specific context, goals, and risk factors
4. Provide well-reasoned financial recommendations based on available information
5. Maintain high ethical standards by:
   - Distinguishing between general and personalized advice
   - Explaining the reasoning behind recommendations
   - Identifying potential risks and trade-offs
   - Acknowledging information limitations
   - Recommending professional consultation when appropriate"""

    financial_advisor_description = "Financial Advisor Agent: A financial advisor that analyzes financial information and market data to provide personalized recommendations. This agent offers thorough analysis while maintaining high ethical standards in financial advising."

    financial_advisor_memory = ListMemory(name="financial_advisor_memory")

    anthropic_client = AnthropicChatCompletion(
        ai_model_id="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY
    )
    settings = AnthropicChatPromptExecutionSettings(temperature=0.0)
    sk_kernel = Kernel(memory=NullMemory())
    claude_3_5_sonnet_client = SKChatCompletionAdapter(
        anthropic_client, kernel=sk_kernel, prompt_settings=settings
    )

    financial_advisor_agent = AssistantAgent(
        name="financial_advisor_agent",
        model_client=claude_3_5_sonnet_client,
        system_message=financial_advisor_system_message,
        description=financial_advisor_description,
        memory=[financial_advisor_memory],
    )

    return financial_advisor_agent


async def test_financial_advisor_agent():
    try:
        agent = await create_financial_advisor_agent()

        test_message = TextMessage(
            content=(
                "I have $50,000 saved and I'm a mid-career professional with moderate risk tolerance. "
                "Considering current market conditions, what diversified investment strategies would you recommend for long-term growth?"
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


async def create_financial_team(
    web_search_agent: AssistantAgent,
    code_writer_agent: AssistantAgent,
    code_executor_agent: CodeExecutorAgent,
    financial_advisor_agent: AssistantAgent,
):
    termination_condition = MaxMessageTermination(max_messages=30)

    final_answer_prompt = """Based on the team's conversation, provide a clear and comprehensive response that addresses the user's question or task. Synthesize insights from web research, calculations, and expert analysis into a natural, well-flowing response. Include relevant data and code results where helpful, explain any important caveats or risks, and ensure complex concepts are explained in accessible language. Keep the tone professional but conversational."""

    team = MagenticOneGroupChat(
        participants=[
            web_search_agent,
            code_writer_agent,
            code_executor_agent,
            financial_advisor_agent,
        ],
        termination_condition=termination_condition,
        model_client=OpenAIChatCompletionClient(model="gpt-4o", timeout=60),
        final_answer_prompt=final_answer_prompt,
    )

    return team


async def test_financial_team():
    try:
        # Agent creation
        web_search_agent = await create_web_search_agent(PERPLEXITY_API_KEY)
        code_writer_agent = await create_code_writer_agent()
        code_executor_agent = await create_code_executor_agent()
        financial_advisor_agent = await create_financial_advisor_agent()

        team = await create_financial_team(
            web_search_agent,
            code_writer_agent,
            code_executor_agent,
            financial_advisor_agent,
        )

        test_message = TextMessage(
            content="Should I invest in VOO?",
            source="user",
        )

        token = CancellationToken()
        print("Sending query to financial team...")

        async for event in team.run_stream(
            task=[test_message], cancellation_token=token
        ):
            display_terminal.pretty_print_event(event)

    except Exception as e:
        logger.exception("Error during financial team test: %s", e)
        raise


async def run_all_tests():
    try:
        # Test perplexity_search function
        logger.info("Testing perplexity_search function...")
        query = "What are the current best practices for retirement savings?"
        result = perplexity_search(query, PERPLEXITY_API_KEY)
        print(result)

        # Test Web Search Agent
        logger.info("Testing Web Search Agent...")
        await test_web_search_agent()

        # Test Code Executor Agent
        logger.info("Testing Code Executor Agent...")
        await test_code_executor_agent()

        # Test Code Writer Agent
        logger.info("Testing Code Writer Agent...")
        await test_code_writer_agent()

        # Test Financial Advisor Agent
        logger.info("Testing Financial Advisor Agent...")
        await test_financial_advisor_agent()

        # Test Financial Team
        logger.info("Testing Financial Team...")
        await test_financial_team()

    except Exception as e:
        logger.exception("Error during tests: %s", e)
        raise


if __name__ == "__main__":
    logger.info("Starting Agent Tests...")
    asyncio.run(run_all_tests())
