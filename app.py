import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from environment.env import ModerationEnv
from environment.models import Action

app = FastAPI(title="AI Content Moderation OpenEnv API")

# Global env instance memory
env_instance = None

class StepRequest(BaseModel):
    action: Action

@app.get("/")
def health_check():
    return {"status": "ok", "environment": "ai-content-moderation"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
