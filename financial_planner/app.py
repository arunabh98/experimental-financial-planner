from typing import AsyncGenerator

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from financial_planner import ANTHROPIC_API_KEY, PERPLEXITY_API_KEY
from financial_planner.agents_team import create_financial_team, format_enhanced_query
from financial_planner.render_utils import stringify_event

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en"
    style="height: auto !important; min-height: auto !important; overflow: auto !important; display: block !important;">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
    <title>Financial Planner</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"
        rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        html,
        body {
            height: auto !important;
            min-height: auto !important;
            overflow: auto !important;
            display: block !important;
            margin: 0;
            padding: 0;
            scroll-behavior: smooth;
        }

        :root {
            --primary-color: #2563eb;
            --secondary-color: #1d4ed8;
            --accent-color: #3b82f6;
            --background-color: #f8fafc;
            --card-background: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --border-color: #e2e8f0;
            --input-bg: #f1f5f9;
            --brand-warning: #f59e0b;
            --brand-info: #0ea5e9;
            --brand-success: #10b981;
            --brand-muted: #64748b;
            --brand-error: #ef4444;
            --warning-bg: #fffbeb;
            --warning-border: #fcd34d;
            --warning-text: #b45309;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        body {
            background: var(--background-color);
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-primary);
            font-size: 15px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .navbar {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 0.8rem 0;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 1030;
            overflow: visible !important;
        }

        .navbar-brand {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff !important;
            letter-spacing: -0.5px;
        }

        .hero {
            position: relative;
            background: linear-gradient(rgba(37, 99, 235, 0.88), rgba(30, 64, 175, 0.92)), url("https://images.unsplash.com/photo-1611095973763-414019e7ick?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
            background-size: cover;
            background-position: center;
            padding: 3.5rem 1rem;
            color: #fff;
            text-align: center;
        }

        .hero h1 {
            font-weight: 800;
            font-size: 2.25rem;
            margin-bottom: 0.75rem;
            letter-spacing: -0.8px;
        }

        .hero p {
            font-size: 1.05rem;
            font-weight: 400;
            max-width: 650px;
            margin: 0 auto;
            opacity: 0.9;
        }

        .main-section {
            padding: 2rem 0;
            position: relative;
            z-index: 1;
        }

        .unified-card,
        .row,
        .col-md-9,
        .col-md-3 {
            overflow: visible !important;
        }

        .unified-card {
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            background: var(--card-background);
            transition: box-shadow 0.3s ease;
            padding: 2rem !important;
        }

        .unified-card h4 {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 1.75rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }

        .form-label {
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .form-control,
        .form-select {
            font-size: 0.95rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 0.85rem 1rem;
            transition: all 0.3s ease;
            background: var(--input-bg);
        }

        .form-control:focus,
        .form-select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
            background: #fff;
        }

        textarea.form-control {
            min-height: 120px;
            resize: vertical;
        }

        .btn-custom {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: #fff;
            border: none;
            padding: 0.85rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            border-radius: 12px;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
        }

        .btn-custom:hover,
        .btn-custom:focus {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(37, 99, 235, 0.25);
            color: #fff;
        }

        .btn-custom:disabled {
            opacity: 0.65;
            cursor: not-allowed;
            transform: none;
            box-shadow: var(--shadow-sm);
        }

        #loading {
            display: none;
        }

        .loading-active {
            display: flex !important;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.8);
            position: absolute;
            inset: 0;
            z-index: 10;
            border-radius: 12px;
        }

        .spinner-border {
            width: 3rem;
            height: 3rem;
            color: var(--primary-color);
        }

        .output-container-wrapper {
            position: relative;
            min-height: 300px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background: #fff;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        #output {
            max-height: 75vh;
            overflow-y: auto !important;
            padding: 1.25rem;
            scroll-behavior: smooth;
        }

        .event-block {
            margin: 0.7rem 0;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: #f9fafb;
            border: 1px solid var(--border-color);
            border-left-width: 5px;
            transition: all 0.2s ease;
            position: relative;
            word-wrap: break-word;
        }

        .event-tool-request {
            border-left-color: var(--brand-warning);
        }

        .event-tool-execution {
            border-left-color: var(--brand-info);
        }

        .event-task-result {
            border-left-color: var(--brand-success);
        }

        .event-text-message {
            border-left-color: var(--brand-muted);
        }

        .event-error {
            border-left-color: var(--brand-error);
            background-color: #fef2f2;
        }

        .event-content {
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text-primary);
            white-space: pre-wrap;
        }

        .event-content pre {
            background-color: #eef2ff;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.85rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }

        .event-content code {
            color: #3730a3;
        }

        .event-block:hover {
            transform: translateX(3px);
            box-shadow: var(--shadow);
            border-color: #d1d5db;
        }

        .event-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .event-meta i {
            font-size: 1rem;
        }

        .bg-danger-custom {
            background-color: var(--brand-error) !important;
            color: #fff !important;
            border-radius: 8px;
            padding: 1rem 1.25rem;
        }

        .text-danger-custom {
            color: var(--brand-error) !important;
        }

        footer {
            background: #e2e8f0;
            padding: 1.5rem 0;
            border-top: 1px solid #cbd5e1;
            margin-top: 3rem;
        }

        footer p {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        footer a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
        }

        footer a:hover {
            text-decoration: underline;
        }

        .disclaimer-alert {
            background-color: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-left: 5px solid var(--brand-warning);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 2rem;
            position: relative;
            box-shadow: var(--shadow-sm);
        }

        .disclaimer-alert h5 {
            color: var(--warning-text);
            font-weight: 700;
            margin-bottom: 0.75rem;
            font-size: 1.05rem;
            display: flex;
            align-items: center;
        }

        .disclaimer-alert h5 i {
            margin-right: 10px;
            color: var(--brand-warning);
            font-size: 1.3rem;
        }

        .disclaimer-alert p {
            color: var(--warning-text);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .disclaimer-checkbox {
            margin-top: 1rem;
            padding: 0.75rem;
            background-color: rgba(255, 255, 255, 0.6);
            border-radius: 8px;
        }

        .disclaimer-checkbox .form-check {
            display: flex;
            align-items: center;
            min-height: 24px;
        }

        .disclaimer-checkbox .form-check-input {
            width: 1.1em;
            height: 1.1em;
            margin-top: 0;
            margin-right: 10px;
            cursor: pointer;
            border: 1px solid var(--warning-text);
        }

        .disclaimer-checkbox .form-check-input:checked {
            background-color: var(--brand-warning);
            border-color: var(--brand-warning);
        }

        .disclaimer-checkbox label {
            font-weight: 500;
            color: var(--warning-text);
            cursor: pointer;
            user-select: none;
            font-size: 0.9rem;
        }

        @media (max-width: 767.98px) {
            .hero {
                padding: 2.5rem 1rem;
            }

            .hero h1 {
                font-size: 1.8rem;
            }

            .hero p {
                font-size: 1rem;
            }

            .main-section {
                padding: 1rem 0;
            }

            .unified-card {
                padding: 1.5rem !important;
            }

            #output {
                max-height: 50vh;
                padding: 1rem;
            }

            .event-block {
                padding: 0.8rem 1rem;
            }

            footer {
                padding: 1rem 0;
                text-align: center;
            }
        }
    </style>
</head>

<body
    style="height:auto !important; min-height:auto !important; overflow:auto !important; display:block !important; margin:0; padding:0;">
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="bi bi-piggy-bank-fill me-2"></i>Financial Planner
            </a>
        </div>
    </nav>

    <div class="hero">
        <div class="container">
            <h1>Personal Financial Planning Assistant</h1>
            <p>Get tailored financial insights and strategies based on your profile and questions.</p>
        </div>
    </div>

    <div class="container-fluid main-section">
        <div class="unified-card p-4 mx-auto" style="max-width: 1400px;">
            <div class="row g-4 g-lg-5">
                <div class="col-lg-4">
                    <h4>Your Profile & Query</h4>

                    <div class="disclaimer-alert" role="alert">
                        <h5><i class="bi bi-exclamation-triangle-fill"></i>Demo Purposes Only</h5>
                        <p>This is a demonstration tool. Information provided is not professional financial advice.</p>
                        <p>Always consult a qualified financial advisor before making decisions.</p>
                        <div class="disclaimer-checkbox">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="disclaimer-agreement" required>
                                <label class="form-check-label" for="disclaimer-agreement">
                                    I understand this is a demo.
                                </label>
                            </div>
                        </div>
                    </div>

                    <form id="chat-form" class="mb-3">
                        <div class="mb-3">
                            <label for="risk_tolerance" class="form-label">Risk Tolerance</label>
                            <select id="risk_tolerance" name="risk_tolerance" class="form-select">
                                <option value="">Not specified</option>
                                <option value="low">Low - Prioritize Safety</option>
                                <option value="moderate">Moderate - Balanced Growth</option>
                                <option value="high">High - Maximize Growth</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label for="time_horizon" class="form-label">Investment Time Horizon</label>
                            <select id="time_horizon" name="time_horizon" class="form-select">
                                <option value="">Not specified</option>
                                <option value="short-term">Short Term (&lt; 2 years)</option>
                                <option value="medium-term">Medium Term (2-5 years)</option>
                                <option value="long-term">Long Term (5+ years)</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label for="annual_gross_income" class="form-label">Approx. Annual Income (USD)</label>
                            <input type="number" id="annual_gross_income" name="annual_gross_income"
                                class="form-control" placeholder="e.g., 75000" min="0">
                        </div>

                        <div class="mb-4">
                            <label for="query" class="form-label">Your Financial Question</label>
                            <textarea id="query" name="query" class="form-control" rows="5"
                                placeholder="e.g., How should I allocate my savings based on my profile?"
                                aria-label="Enter your financial question"></textarea>
                        </div>

                        <button type="submit" class="btn btn-custom w-100" aria-label="Send your financial query"
                            id="submit-btn">
                            <i class="bi bi-send-fill me-2"></i>Get Analysis
                        </button>
                        <div id="form-error" class="text-danger-custom mt-3" role="alert" aria-live="assertive"></div>
                    </form>
                </div>
                <div class="col-lg-8">
                    <h4>Analysis & Recommendations</h4>
                    <div class="output-container-wrapper">
                        <div id="loading" role="status" aria-live="polite">
                            <div class="spinner-border" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-3 text-muted">Processing your request, please wait...</p>
                        </div>
                        <div id="output" aria-live="polite">
                            <p class="text-secondary p-3">Your analysis will appear here once you submit a query.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="text-center">
        <div class="container">
            <p>
                Created by
                <a href="https://github.com/arunabh98" target="_blank" rel="noopener noreferrer">Arunabh Ghosh</a>
            </p>
            <p class="mt-2" style="font-size: 0.85rem;">Disclaimer: This tool is provided for research purposes only and
                does not constitute professional financial advice. Always consult a qualified financial advisor.</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const queryEl = document.getElementById('query');
        const form = document.getElementById('chat-form');
        const disclaimerCheckbox = document.getElementById('disclaimer-agreement');
        const submitBtn = document.getElementById('submit-btn');
        const outputEl = document.getElementById('output');
        const loadingEl = document.getElementById('loading');
        const formErrorEl = document.getElementById('form-error');

        queryEl.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.max(120, this.scrollHeight) + 'px';
        }, false);

        function validateForm() {
            formErrorEl.textContent = '';
            const query = queryEl.value.trim();

            if (!disclaimerCheckbox.checked) {
                formErrorEl.textContent = 'Please acknowledge the disclaimer before submitting.';
                disclaimerCheckbox.focus();
                return false;
            }
            if (!query) {
                formErrorEl.textContent = 'Please enter your financial question before submitting.';
                queryEl.focus();
                return false;
            }
            return true;
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!validateForm()) {
                return;
            }

            const query = queryEl.value.trim();
            const riskTolerance = document.getElementById('risk_tolerance').value;
            const timeHorizon = document.getElementById('time_horizon').value;
            const annualGrossIncome = document.getElementById('annual_gross_income').value;

            outputEl.innerHTML = "";
            formErrorEl.textContent = '';
            loadingEl.classList.add('loading-active');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Processing...';

            try {
                const response = await fetch('/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        risk_tolerance: riskTolerance || null,
                        time_horizon: timeHorizon || null,
                        annual_gross_income: annualGrossIncome || null
                    })
                });

                if (!response.ok) {
                    let errorMsg = `HTTP Error: ${response.status} ${response.statusText}`;
                    try {
                        const errData = await response.json();
                        errorMsg += ` - ${errData.detail || 'Unknown server error'}`;
                    } catch (jsonError) { }
                    throw new Error(errorMsg);
                }

                if (!response.body || typeof response.body.getReader !== 'function') {
                    const text = await response.text();
                    outputEl.innerHTML = text;
                    loadingEl.classList.remove('loading-active');
                    console.warn("Response was not streamable.");
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let firstChunk = true;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    if (firstChunk) {
                        loadingEl.classList.remove('loading-active');
                        firstChunk = false;
                    }

                    const chunk = decoder.decode(value, { stream: true });
                    outputEl.insertAdjacentHTML('beforeend', chunk);
                    outputEl.scrollTop = outputEl.scrollHeight;
                }

            } catch (err) {
                console.error("Fetch Error:", err);
                loadingEl.classList.remove('loading-active');
                outputEl.innerHTML = `<div class='event-block event-error'>
                                         <div class='event-meta'><i class='bi bi-exclamation-octagon-fill me-2'></i>Error</div>
                                         <div class='event-content'>Failed to get analysis: ${err.message || err}. Please try again later.</div>
                                      </div>`;
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="bi bi-send-fill me-2"></i>Get Analysis';
            }
        });
    </script>
</body>

</html>
"""


@app.post("/infer")
async def infer(request: Request):
    code_executor = None
    try:
        body = await request.json()
        query = body.get("query", "").strip()

        risk_tolerance = body.get("risk_tolerance") or None
        time_horizon = body.get("time_horizon") or None
        annual_gross_income_str = body.get("annual_gross_income") or None
        annual_gross_income = None
        if annual_gross_income_str:
            try:
                annual_gross_income = float(annual_gross_income_str)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid 'annual_gross_income' format."
                )

        if not query:
            raise HTTPException(
                status_code=400, detail="Missing or empty 'query' in request."
            )

        enhanced_query = format_enhanced_query(
            query,
            risk_tolerance,
            time_horizon,
            annual_gross_income,
        )

        if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY == "YOUR_PERPLEXITY_API_KEY":
            raise HTTPException(
                status_code=500, detail="Perplexity API key not configured."
            )
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY":
            raise HTTPException(
                status_code=500, detail="Anthropic API key not configured."
            )

        team, code_executor = await create_financial_team(
            perplexity_api_key=PERPLEXITY_API_KEY,
            anthropic_api_key=ANTHROPIC_API_KEY,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon,
            annual_gross_income=annual_gross_income,
        )

        async def event_generator() -> AsyncGenerator[str, None]:
            cancellation_token = CancellationToken()
            try:
                async for event in team.run_stream(
                    task=[TextMessage(content=enhanced_query, source="user")],
                    cancellation_token=cancellation_token,
                ):
                    if (
                        isinstance(event, TextMessage)
                        and getattr(event, "source", None) == "user"
                    ):
                        continue

                    if event.__class__.__name__ == "MemoryQueryEvent":
                        continue

                    stringified = stringify_event(event)
                    if stringified:
                        yield stringified

            except Exception as e:
                yield f"""<div class='event-block event-error'>
                             <div class='event-meta'><i class='bi bi-exclamation-octagon-fill me-2'></i>Processing Error</div>
                             <div class='event-content'>An error occurred during analysis: {str(e)}</div>
                          </div>"""

            finally:
                if code_executor:
                    await code_executor.stop()

        return StreamingResponse(event_generator(), media_type="text/html")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {str(e)}"
        )
