from typing import AsyncGenerator

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from financial_planner import PERPLEXITY_API_KEY
from financial_planner.agents_team import (
    create_code_executor_agent,
    create_code_writer_agent,
    create_financial_team,
    create_web_search_agent,
)
from financial_planner.render_utils import stringify_event

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Financial Planner</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" />
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --accent-color: #3b82f6;
            --background-color: #f8fafc;
            --card-background: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --brand-warning: #f59e0b;
            --brand-info: #0ea5e9;
            --brand-success: #10b981;
            --brand-muted: #6b7280;
            --brand-error: #dc2626;
        }
        
        body {
            margin: 0;
            padding: 0;
            background: var(--background-color);
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-primary);
            font-size: 15px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        .navbar {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .navbar-brand {
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff !important;
            letter-spacing: -0.5px;
        }

        .hero {
            position: relative;
            background: linear-gradient(
                rgba(37, 99, 235, 0.9),
                rgba(30, 64, 175, 0.95)
            ),
            url("https://images.unsplash.com/photo-1581091870628-1d29c2701e48?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
            background-size: cover;
            background-position: center;
            padding: 3rem 1rem;
            color: #fff;
            text-align: center;
        }

        .hero h1 {
            font-weight: 800;
            font-size: 2rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            animation: fadeInDown 1s ease forwards;
            opacity: 0;
        }

        .hero p {
            font-size: 1rem;
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
            opacity: 0;
            animation: fadeInUp 1s ease forwards 0.3s;
        }

        .main-section {
            padding: 1rem 0;
            position: relative;
            z-index: 1;
        }

        .unified-card {
            border: none;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            background: var(--card-background);
            transition: all 0.3s ease;
        }

        .unified-card h4 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
        }

        textarea.form-control {
            font-size: 0.95rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            transition: all 0.3s ease;
            background: #f8fafc;
        }

        textarea.form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            background: #fff;
        }

        .btn-custom {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: #fff;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border-radius: 12px;
            transition: all 0.3s ease;
        }

        .btn-custom:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }

        .output-container {
            min-height: 500px;
            max-height: 700px;
            overflow-y: auto;
            background: #fff;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
        }

        #loading {
            min-height: 500px;
            max-height: 700px;
            overflow-y: auto;
            background: #fff;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
            display: none;
        }

        .loading-active {
            display: flex !important;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .event-block {
            margin: 1rem 0;
            padding: 1.25rem;
            border-radius: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
            position: relative;
        }

        .event-block::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            border-radius: 12px 0 0 12px;
        }

        .event-tool-request::before {
            background-color: var(--brand-warning);
        }

        .event-tool-execution::before {
            background-color: var(--brand-info);
        }

        .event-task-result::before {
            background-color: var(--brand-success);
        }

        .event-text-message::before {
            background-color: var(--brand-muted);
        }

        .event-content {
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--text-primary);
            white-space: pre-wrap;
        }

        .event-block:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .event-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .bg-danger {
            background-color: var(--brand-error) !important;
            color: #fff !important;
        }

        footer {
            background: #f1f5f9;
            padding: 1rem 0;
            border-top: 1px solid #e2e8f0;
        }

        footer p {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="bi bi-piggy-bank me-2"></i>Financial Planner
            </a>
        </div>
    </nav>

    <div class="hero">
        <h1>Personal Financial Planning Assistant</h1>
        <p>Get expert financial advice, personalized strategies, and actionable insights to secure your financial future.</p>
    </div>

    <div class="container-fluid main-section">
        <div class="unified-card p-4 mx-auto" style="max-width: none;">
            <div class="row g-4">
                <div class="col-md-3">
                    <h4>Ask Your Financial Questions</h4>
                    <form id="chat-form" class="mb-3">
                        <div class="mb-4">
                            <textarea
                                id="query"
                                name="query"
                                class="form-control"
                                rows="5"
                                placeholder="Type your financial question here..."
                                aria-label="Enter your financial question"
                            ></textarea>
                        </div>
                        <button type="submit" class="btn btn-custom" aria-label="Send your financial query">
                            <i class="bi bi-send me-2"></i>Send Query
                        </button>
                    </form>
                </div>
                <div class="col-md-9">
                    <h4>Analysis & Recommendations</h4>
                    <div id="loading" class="text-center p-4" style="display: none;">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2 text-muted">Processing your request...</p>
                    </div>
                    <div id="output" class="output-container" aria-live="polite"></div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <div class="container">
            <p class="mb-0">Created by <a href="https://github.com/arunabh98" target="_blank" rel="noopener noreferrer" style="color: var(--primary-color); text-decoration: none; font-weight: 500;">Arunabh Ghosh</a></p>
            <p class="mt-2 text-muted" style="font-size: 0.85rem;">
                Disclaimer: This tool is provided for research purposes only and does not constitute professional financial advice. Always consult a qualified financial advisor before making any financial decisions.
            </p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        const queryEl = document.getElementById('query');
        queryEl.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    </script>

    <script>
        const form = document.getElementById('chat-form');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = queryEl.value.trim();
            const output = document.getElementById('output');
            const loading = document.getElementById('loading');
            output.innerHTML = "";

            if (!query) {
                output.innerHTML = "<div class='text-danger'>Please enter a query before submitting.</div>";
                return;
            }

            loading.classList.add('loading-active');
            output.style.display = 'none';

            try {
                const response = await fetch('/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });

                if (!response.ok) {
                    loading.classList.remove('loading-active');
                    output.style.display = 'block';
                    output.innerHTML = `<div class='text-danger'>HTTP Error: ${response.status} ${response.statusText}</div>`;
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let firstEvent = true;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    if (firstEvent) {
                        loading.classList.remove('loading-active');
                        output.style.display = 'block';
                        firstEvent = false;
                    }
                    
                    output.innerHTML += decoder.decode(value);
                    output.scrollTop = output.scrollHeight;
                }
            } catch (err) {
                loading.classList.remove('loading-active');
                output.style.display = 'block';
                output.innerHTML = `<div class='text-danger'>Error: ${err}</div>`;
            }
        });
    </script>
</body>
</html>"""


@app.post("/infer")
async def infer(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            raise HTTPException(
                status_code=400, detail="Missing 'query' in request JSON."
            )

        web_search_agent = await create_web_search_agent(PERPLEXITY_API_KEY)
        code_writer_agent = await create_code_writer_agent()
        code_executor_agent = await create_code_executor_agent()
        team = await create_financial_team(
            web_search_agent, code_writer_agent, code_executor_agent
        )

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in team.run_stream(
                    task=[TextMessage(content=query, source="user")],
                    cancellation_token=CancellationToken(),
                ):
                    yield stringify_event(event)
            except Exception as e:
                yield f"<div class='event-block bg-danger text-white'><strong>ERROR:</strong> {str(e)}</div>"

        return StreamingResponse(event_generator(), media_type="text/html")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
