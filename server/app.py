import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

from environment.env import ModerationEnv
from environment.models import Action
from inference import run_inference

from environment.env import ModerationEnv
from environment.models import Action

app = FastAPI(title="AI Content Moderation OpenEnv API")

# Global env instance memory
env_instance = None

class StepRequest(BaseModel):
    action: Action

@app.get("/health")
def health_check():
    return {"status": "ok", "environment": "ai-content-moderation"}

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Content Moderation OpenEnv</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --bg: #0f172a;
                --surface: #1e293b;
                --text: #f8fafc;
                --accent: #10b981;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }
            header {
                text-align: center;
                padding: 4rem 2rem 2rem;
                background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(15,23,42,1) 100%);
                width: 100%;
                border-bottom: 1px solid #334155;
            }
            h1 {
                font-size: 3rem;
                font-weight: 800;
                margin: 0;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p.subtitle {
                font-size: 1.2rem;
                color: #94a3b8;
                max-width: 600px;
                margin: 1rem auto;
                line-height: 1.6;
            }
            .container {
                max-width: 800px;
                width: 90%;
                margin: 3rem auto;
            }
            .card {
                background: var(--surface);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
                border: 1px solid #334155;
                transition: transform 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .api-docs {
                background: #0b1120;
                padding: 1.5rem;
                border-radius: 12px;
                font-family: monospace;
                color: #38bdf8;
                margin-top: 1rem;
                white-space: pre-wrap;
            }
            button {
                background: var(--primary);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                margin-top: 1rem;
            }
            button:hover {
                background: #4f46e5;
                box-shadow: 0 0 15px rgba(99,102,241,0.5);
            }
            .loading {
                display: none;
                margin-top: 1rem;
                color: #94a3b8;
            }
            .result-box {
                display: none;
                margin-top: 2rem;
                background: #0f172a;
                border-left: 4px solid var(--accent);
                padding: 1.5rem;
                border-radius: 0 8px 8px 0;
            }
            .endpoint-tag {
                background: #334155;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
                color: #f1f5f9;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>AI Moderation Env</h1>
            <p class="subtitle">A rigorous reinforcement learning environment for testing and evaluating LLM safety protocols.</p>
        </header>

        <div class="container">
            <div class="card">
                <h2>🚀 Live Agent Evaluation</h2>
                <p style="color: #94a3b8;">Trigger your AI agent to solve the environment perfectly using the configured OpenAI API key.</p>
                <div>
                    <select id="taskSelect" style="padding: 10px; border-radius: 8px; background: #0f172a; color: white; border: 1px solid #334155;">
                        <option value="task_easy">Easy Task (Spam Detection)</option>
                        <option value="task_medium">Medium Task (Toxicity)</option>
                        <option value="task_hard">Hard Task (Contextual/Thread)</option>
                    </select>
                    <button onclick="runEvaluation()">Run AI Inference Loop</button>
                    <div id="loading" class="loading">⏳ Running benchmark with LLM... (this may take up to a minute)</div>
                    <div id="result" class="result-box"></div>
                </div>
            </div>

            <div class="card" style="margin-top: 2rem;">
                <h2>⚙️ Available API Endpoints</h2>
                <div style="margin-top: 1rem;">
                    <p><span class="endpoint-tag">POST</span> <b>/reset</b> - Grabs a new observation state</p>
                    <p><span class="endpoint-tag">POST</span> <b>/step</b> - Submit a moderation action</p>
                    <p><span class="endpoint-tag">POST</span> <b>/evaluate</b> - Run full baseline dataset</p>
                    <p><span class="endpoint-tag">GET</span> <b>/docs</b> - View interactive Swagger UI</p>
                </div>
            </div>
        </div>

        <script>
            async function runEvaluation() {
                const btn = document.querySelector('button');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                const task = document.getElementById('taskSelect').value;
                
                btn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';
                
                try {
                    const response = await fetch(`/evaluate?task_id=${task}`, { method: 'POST' });
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to evaluate');
                    }
                    
                    result.innerHTML = `
                        <h3 style="margin-top:0; color:#10b981;">Evaluation Complete! 🏆</h3>
                        <p><b>Model:</b> ${data.model}</p>
                        <p><b>Final Score:</b> ${(data.average_reward * 100).toFixed(2)}%</p>
                        <p><b>Correct Moderations:</b> ${data.correct_moderations}</p>
                        <p><b>False Positives:</b> ${data.false_positives}</p>
                        <p><b>Missed Harmful:</b> ${data.missed_harmful_content}</p>
                    `;
                    result.style.display = 'block';
                } catch (err) {
                    result.innerHTML = `<h3 style="margin-top:0; color:#ef4444;">Error</h3><p>${err.message}</p>
                    <p style="font-size:0.9rem; color:#94a3b8;">Did you add OPENAI_API_KEY to your Space Secrets?</p>`;
                    result.style.display = 'block';
                } finally {
                    btn.disabled = false;
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/reset")
def reset_environment(task_id: str = "task_medium"):
    global env_instance
    try:
        env_instance = ModerationEnv(task_id=task_id)
        obs = env_instance.reset()
        state = env_instance.state()
        return {
            "observation": obs.model_dump(),
            "state": state.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_environment(req: StepRequest):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        obs, reward, state, done, info = env_instance.step(req.action)
        return {
            "observation": obs.model_dump() if obs else None,
            "reward": reward.model_dump(),
            "state": state.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate")
def evaluate_agent(task_id: str = "task_medium"):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY secret is not configured in Hugging Face!")
    
    try:
        # Run the baseline agent and return its final metrics
        result = run_inference(task_id=task_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def serve():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    serve()
