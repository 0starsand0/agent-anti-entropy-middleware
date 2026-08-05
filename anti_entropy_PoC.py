# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Anti-Entropy Core Middleware PoC
Author: Starsand
Jurisdiction: HKSAR (Hong Kong Special Administrative Region)
License: PolyForm Noncommercial License 1.0.0 (Non-Commercial Use Only)

This software is licensed under the PolyForm Noncommercial License 1.0.0.
You may use, modify, and distribute this software for non-commercial purposes only.
Commercial use, embedding in commercial products, or operating as a commercial 
service is strictly prohibited without prior written authorization from the author.
"""

import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

app = FastAPI(title="Anti-Entropy Core Middleware PoC", version="1.0.0")

# ==========================================
# 1. PID Controller Implementation
# ==========================================
class PIDController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05, target_entropy=2.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_entropy = target_entropy
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def compute(self, current_metric: float) -> float:
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-6

        error = self.target_entropy - current_metric
        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.last_error = error
        self.last_time = now
        return output

pid = PIDController()

# ==========================================
# 2. Hybrid Loop & Homeostatic Detection
# ==========================================
def calculate_ngram_repetition(text: str, n: int = 3) -> float:
    """Calculate N-gram repetition ratio (R_ngram)"""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def evaluate_homeostatic_load(messages: list) -> float:
    """Evaluate system homeostatic load to trigger PID adjustments during loops"""
    if not messages:
        return 2.5
    last_content = messages[-1].get("content", "")
    ngram_rep = calculate_ngram_repetition(last_content, n=3)
    
    # If high repetition is detected, drop homeostatic load to trigger thermal breakout
    if ngram_rep > 0.4:
        return 0.2  
    return 2.5      # Normal equilibrium

# ==========================================
# 3. FastAPI Proxy & Dynamic Temperature Interceptor
# ==========================================
BACKEND_URL = "http://localhost:8080/v1/chat/completions"

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    
    messages = body.get("messages", [])
    homeo_load = evaluate_homeostatic_load(messages)
    
    # Dynamic temperature computation via PID
    base_temp = body.get("temperature", 0.7)
    delta_t = pid.compute(homeo_load)
    adjusted_temp = max(0.1, min(2.0, base_temp + delta_t))
    body["temperature"] = adjusted_temp
    
    print(f"[Anti-Entropy Middleware] Base Temp: {base_temp} -> Adjusted: {adjusted_temp:.4f}")
    
    client = httpx.AsyncClient(timeout=60.0)
    try:
        req = client.build_request("POST", BACKEND_URL, json=body, headers=dict(request.headers))
        response = await client.send(req, stream=True)
    except httpx.RequestError:
        # Fallback mock response for local testing if backend is offline
        return JSONResponse({
            "id": "mock-completion",
            "object": "chat.completion",
            "choices": [{
                "index": 0, 
                "message": {
                    "role": "assistant", 
                    "content": f"[Anti-Entropy Core] Temperature dynamically modulated to {adjusted_temp:.4f}."
                }
            }]
        })
    
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=dict(response.headers)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)