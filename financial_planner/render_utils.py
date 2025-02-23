import json

call_id_map = {}
css_added = False


EVENT_TYPE_DISPLAY_NAMES = {
    "TextMessage": "Message",
    "ToolCallRequestEvent": "Tool Request",
    "ToolCallExecutionEvent": "Tool Response",
    "TaskResult": "Final Result",
}


SOURCE_DISPLAY_NAMES = {
    "user": "You",
    "MagenticOneOrchestrator": "Team Coordinator",
    "financial_advisor_agent": "Financial Advisor",
    "web_search_agent": "Market Researcher",
    "code_writer_agent": "Code Writer",
    "code_executor_agent": "Code Executor",
}


def get_base_css() -> str:
    return """
        <style>
            :root {
                --brand-warning: #f59e0b;
                --brand-info: #0ea5e9;
                --brand-success: #10b981;
                --brand-muted: #6b7280;
                --brand-error: #dc2626;
                --brand-blue-light: #eff6ff;
                --brand-blue-lighter: #dbeafe;
                --brand-green-light: #f0fdf4;
                --brand-green-lighter: #dcfce7;
                --brand-purple-light: #f5f3ff;
                --brand-purple-lighter: #ede9fe;
                --brand-orange-light: #fff7ed;
                --brand-orange-lighter: #ffedd5;
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --card-background: #ffffff;
            }
            .event-block {
                margin: 15px 0;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                background: #ffffff;
                background: var(--card-background);
                opacity: 1;
                color: #1e293b;
                color: var(--text-primary);
            }
            .event-block:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            .gradient-bg-blue {
                background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                background: linear-gradient(135deg, var(--brand-blue-light) 0%, var(--brand-blue-lighter) 100%);
            }
            .gradient-bg-green {
                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                background: linear-gradient(135deg, var(--brand-green-light) 0%, var(--brand-green-lighter) 100%);
            }
            .gradient-bg-purple {
                background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
                background: linear-gradient(135deg, var(--brand-purple-light) 0%, var(--brand-purple-lighter) 100%);
            }
            .gradient-bg-orange {
                background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
                background: linear-gradient(135deg, var(--brand-orange-light) 0%, var(--brand-orange-lighter) 100%);
            }
            .event-tool-request::before,
            .event-tool-execution::before,
            .event-task-result::before,
            .event-text-message::before {
                content: "";
                position: absolute;
                left: 0;
                top: 0;
                bottom: 0;
                width: 5px;
                border-radius: 12px 0 0 12px;
            }
            .event-tool-request::before {
                background-color: #f59e0b;
                background-color: var(--brand-warning);
            }
            .event-tool-execution::before {
                background-color: #0ea5e9;
                background-color: var(--brand-info);
            }
            .event-task-result::before {
                background-color: #10b981;
                background-color: var(--brand-success);
            }
            .event-text-message::before {
                background-color: #6b7280;
                background-color: var(--brand-muted);
            }
            .event-meta {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
                font-size: 14px;
                color: #64748b;
                color: var(--text-secondary);
            }
            .event-icon {
                margin-right: 10px;
                font-size: 20px;
            }
            .event-type {
                font-weight: 600;
                color: #1e293b;
                color: var(--text-primary);
            }
            .event-content {
                background: rgba(255, 255, 255, 0.7);
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
            }
            .rotating-gear {
                animation: rotate 4s linear infinite;
            }
            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            .pulse-animation {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            .success-animation {
                animation: success 0.5s ease-in;
            }
            @keyframes success {
                0% { transform: scale(0); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }
            .bounce-animation {
                animation: bounce 1s infinite;
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
            .tool-call, .execution-result, .dict-item, .list-item {
                margin: 8px 0;
            }
            .query-text {
                margin-top: 5px;
                color: #64748b;
                color: var(--text-secondary);
            }
            .success-text {
                color: #10b981;
                color: var(--brand-success);
                margin: 5px 0;
            }
            .final-answer {
                background: rgba(255, 255, 255, 0.9);
                padding: 12px;
                border-radius: 6px;
            }
            .stop-reason {
                margin-top: 10px;
                color: #64748b;
                color: var(--text-secondary);
            }
            .no-answer {
                color: #64748b;
                color: var(--text-secondary);
                font-style: italic;
            }
        </style>
    """


def stringify_event(event: object) -> str:
    global css_added
    event_type = event.__class__.__name__
    source = getattr(event, "source", None)
    target = getattr(event, "target", None)
    content = getattr(event, "content", None)

    css_class = "event-block"
    icon_html = "<i class='bi bi-info-circle pulse-animation'></i>"

    if event_type == "ToolCallRequestEvent":
        css_class += " event-tool-request gradient-bg-blue"
        icon_html = "<i class='bi bi-gear rotating-gear'></i>"
    elif event_type == "ToolCallExecutionEvent":
        css_class += " event-tool-execution gradient-bg-green"
        icon_html = "<i class='bi bi-tools tool-animation'></i>"
    elif event_type == "TaskResult":
        css_class += " event-task-result gradient-bg-purple"
        icon_html = "<i class='bi bi-check-circle success-animation'></i>"
    elif event_type == "TextMessage":
        css_class += " event-text-message gradient-bg-orange"
        icon_html = "<i class='bi bi-chat-left-text bounce-animation'></i>"

    html = []
    if not css_added:
        html.append(get_base_css())
        css_added = True

    html.append(f"<div class='{css_class}' style='position: relative;'>")
    html.append("<div class='event-meta'>")
    html.append(f"<span class='event-icon'>{icon_html}</span>")
    friendly_name = EVENT_TYPE_DISPLAY_NAMES.get(event_type, event_type)
    html.append(f"<span class='event-type'>{escape_html(friendly_name)}</span>")
    html.append("</div>")

    if source:
        friendly_source = SOURCE_DISPLAY_NAMES.get(source, source)
        html.append(
            f"<div class='event-meta'><span>From:</span> {escape_html(friendly_source)}</div>"
        )
    if target:
        html.append(
            f"<div class='event-meta'><span>Target:</span> {escape_html(target)}</div>"
        )

    if event_type == "TaskResult":
        html.append("<div class='event-content'>")
        html.append(render_task_result(event))
        html.append("</div></div>")
        return "".join(html)

    if content is not None:
        html.append("<div class='event-content'>")
        if event_type == "ToolCallRequestEvent":
            html.append(render_tool_call_request_event(content))
        elif event_type == "ToolCallExecutionEvent":
            html.append(render_tool_call_execution_event(content))
        else:
            html.append(render_content(content))
        html.append("</div>")

    html.append("</div>")
    return "".join(html)


def render_tool_call_request_event(content: list) -> str:
    lines = []
    for call in content:
        function_name = getattr(call, "name", "unknown_tool")
        call_id = getattr(call, "id", "")
        raw_args = getattr(call, "arguments", "{}")

        try:
            args_dict = json.loads(raw_args)
            query_str = args_dict.get("query", "")
        except Exception:
            query_str = raw_args

        call_id_map[call_id] = (function_name, query_str)
        lines.append(
            f"<div class='tool-call'><strong>{escape_html(function_name)}</strong> "
            f"<div class='query-text'>Query: <em>{escape_html(query_str)}</em></div></div>"
        )
    return "<br>".join(lines)


def render_tool_call_execution_event(content: list) -> str:
    lines = []
    for result in content:
        call_id = getattr(result, "call_id", "")
        tool_name, original_query = call_id_map.get(call_id, ("unknown_tool", ""))
        lines.append(
            f"<div class='execution-result'><strong>{escape_html(tool_name)}</strong> "
            f"<div class='success-text'>✓ Completed successfully</div>"
            f"<div class='query-text'>Query: <em>{escape_html(original_query)}</em></div></div>"
        )
    return "<br>".join(lines)


def render_task_result(event: object) -> str:
    stop_reason = getattr(event, "stop_reason", None)
    messages = getattr(event, "messages", None)

    if not messages:
        possible_msg_list = getattr(event, "content", None)
        messages = possible_msg_list if isinstance(possible_msg_list, list) else []

    final_answer = None
    for m in reversed(messages):
        if (
            m.__class__.__name__ in ("TextMessage", "ToolCallSummaryMessage")
            and getattr(m, "source", "user") != "user"
        ):
            potential_answer = getattr(m, "content", "")
            if potential_answer:
                final_answer = potential_answer
                break

    rendered = []
    if final_answer:
        rendered.append(
            f"<div class='final-answer'><strong>Final Answer:</strong><div class='answer-content'>{escape_html(final_answer)}</div></div>"
        )
    else:
        rendered.append(
            "<div class='no-answer'><em>No final answer was provided by the agent.</em></div>"
        )

    if stop_reason:
        rendered.append(
            f"<div class='stop-reason'><em>Reason: {escape_html(stop_reason)}</em></div>"
        )

    return "<br>".join(rendered)


def render_content(content) -> str:
    if isinstance(content, str):
        return escape_html(content)
    if isinstance(content, list):
        return "".join(
            f"<div class='list-item'>• {escape_html(repr(item) if not isinstance(item, str) else item)}</div>"
            for item in content
        )
    if isinstance(content, dict):
        return "".join(
            f"<div class='dict-item'><strong>{escape_html(k)}:</strong> "
            f"{render_content(v) if not isinstance(v, str) else escape_html(v)}</div>"
            for k, v in content.items()
        )
    return escape_html(repr(content))


def escape_html(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
