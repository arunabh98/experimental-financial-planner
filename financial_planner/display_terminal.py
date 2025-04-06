import logging
import re

from autogen_agentchat.messages import (
    MemoryQueryEvent,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import FunctionCall
from autogen_core.memory import MemoryContent
from autogen_core.models import FunctionExecutionResult
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

logger = logging.getLogger(__name__)
console = Console()


def pretty_print_event(event: object) -> None:
    event_type = event.__class__.__name__
    header = f"[bold blue]Event: {event_type}"

    if hasattr(event, "source"):
        header += f" [Source: {event.source}]"
    if hasattr(event, "target"):
        header += f" [Target: {event.target}]"

    content_parts = (
        [f"Stop Reason: {event.stop_reason}"]
        + [line for message in event.messages for line in format_message(message)]
        if event_type == "TaskResult"
        else format_generic_event(event)
    )

    rendered_content = "\n".join(content_parts)

    if "```" in rendered_content and event_type != "TaskResult":
        try:
            match = re.search(r"```(\w+)\s*(.*?)```", rendered_content, re.DOTALL)
            if match and match.group(2).strip():
                rendered_content = Syntax(
                    match.group(2).strip(),
                    match.group(1),
                    theme="monokai",
                    line_numbers=True,
                )
        except Exception as e:
            logger.warning(f"Code block parsing failed for {event_type}: {e}")

    console.print(
        Panel(rendered_content, title=header, expand=(event_type == "TaskResult"))
    )


def format_message(message: object) -> list[str]:
    lines = []
    message_info = f"  - {getattr(message, 'source', 'Unknown Source')} ({message.__class__.__name__})"
    if hasattr(message, "models_usage") and message.models_usage:
        message_info += f" Models Usage: {message.models_usage}"
    lines.append(message_info)

    if isinstance(message, TextMessage):
        lines.append(f"    Content: {message.content}")
    elif isinstance(message, ToolCallRequestEvent):
        for call in message.content:
            lines.extend(
                [
                    f"    - Function Call: {call.name} (ID: {call.id})",
                    f"      Arguments: {call.arguments}",
                ]
            )
    elif isinstance(message, (ToolCallExecutionEvent, ToolCallSummaryMessage)):
        if isinstance(message, ToolCallExecutionEvent):
            lines.extend(
                f"    - Execution Result (Call ID: {result.call_id}): {result.content}"
                for result in message.content
            )
        else:
            lines.append(f"    Content: {message.content}")
    elif isinstance(message, (MemoryQueryEvent)):
        lines.append(f"    Memory Operation: {message.__class__.__name__}")
        memory_content = getattr(message, "content", [])
        for item in memory_content:
            lines.append(format_memory_content(item))
    else:
        logger.warning(f"Unknown message type in TaskResult: {type(message)}")
        lines.extend(
            [
                f"    [red]Unknown message type: {type(message)}[/red]",
                f"    Details: {format_generic_event(message)}",
            ]
        )
    return lines


def format_memory_content(item):
    if isinstance(item, MemoryContent):
        return f"    - Memory Content: {item.content[:100]}{'...' if len(item.content) > 100 else ''} (Type: {item.mime_type})"
    return f"    - {item}"


def format_generic_event(event: object) -> list[str]:
    lines = []
    if hasattr(event, "content"):
        content = event.content
        if isinstance(content, str):
            lines.append(f"  Content: {content}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    lines.append(f"  - {item}")
                elif isinstance(item, FunctionCall):
                    lines.extend(
                        [
                            f"    - Function Call: {item.name} (ID: {item.id})",
                            f"      Arguments: {item.arguments}",
                        ]
                    )
                elif isinstance(item, FunctionExecutionResult):
                    lines.append(
                        f"    - Execution Result (Call ID: {item.call_id}): {item.content}"
                    )
                elif isinstance(item, MemoryContent):
                    lines.append(format_memory_content(item))
                else:
                    logger.warning(f"Unknown content list item type: {type(item)}")
                    lines.append(f"    - Unknown item: {item}")
        elif isinstance(content, dict):
            lines.extend(f"   {key}: {value}" for key, value in content.items())
        else:
            logger.warning(f"Unknown content type: {type(content)}")
            lines.append(f"    Content: {content}")

    for key, value in event.__dict__.items():
        if key not in ("content", "source", "target"):
            lines.append(f"  {key}: {value}")

    if not lines:
        logger.warning(f"No displayable attributes for event type: {type(event)}")
        lines.append(f"  Details: {str(event)}")

    return lines
