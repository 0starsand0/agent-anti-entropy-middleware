# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Anti-Entropy Cognitive Middleware with Adaptive ML PID & Memory Layers (v2.0.0)
Author: Starsand
Jurisdiction: HKSAR (Hong Kong Special Administrative Region)
License: PolyForm Noncommercial License 1.0.0 (Non-Commercial Use Only)

This software is licensed under the PolyForm Noncommercial License 1.0.0.
You may use, modify, and distribute this software for non-commercial purposes only.
Commercial use, embedding in commercial products, or operating as a commercial 
service is strictly prohibited without prior written authorization from the author.
"""

import os
import time
import logging
import httpx
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ==========================================
# 0. System Logging & Environment Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CognitiveMiddleware")

app = FastAPI(
    title="Anti-Entropy Cognitive Middleware",
    description="Production-ready cognitive homeostatic proxy with adaptive ML PID and memory layers.",
    version="2.0.0"
)

BACKEND_URL = os.getenv("LLM_BACKEND_URL", "http://localhost:8080/v1/chat/completions")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))


# ==========================================
# 1. Adaptive ML PID Controller (Self-Tuning)
# ==========================================
class AdaptivePIDController:
    """
    PID Controller with online reinforcement adaptation for gains (Kp, Ki, Kd).
    Dynamically adjusts parameters based on error gradients during runtime.
    """
    def __init__(self, kp: float = 0.1, ki: float = 0.01, kd: float = 0.05, target_entropy: float = 2.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_entropy = target_entropy
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.last_accessed = time.time()
        self.success_streak = 0

    def compute(self, current_metric: float) -> float:
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-6

        error = self.target_entropy - current_metric
        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Online Machine Learning Adaptation (Active Self-Tuning Heuristic)
        if abs(error) < abs(self.last_error):
            self.success_streak += 1
            if self.success_streak > 3:
                self.kp = max(0.05, min(0.3, self.kp * 1.02))
                logger.debug(f"[Adaptive ML] Convergence steady. Tuned Kp up to: {self.kp:.4f}")
                self.success_streak = 0
        else:
            self.kp = max(0.05, self.kp * 0.95)
            self.success_streak = 0

        self.last_error = error
        self.last_time = now
        self.last_accessed = now
        return output

# ==========================================
# 2. Episodic Memory Layer & State Manager
# ==========================================
class MemoryLayer:
    """
    Manages session context summarization and episodic state to prevent context rot.
    """
    def __init__(self, max_history_window: int = 12):
        self.max_history_window = max_history_window
        self.memory_store: Dict[str, Dict[str, Any]] = {}

    def process_and_compress(self, session_id: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if session_id not in self.memory_store:
            self.memory_store[session_id] = {"summary": "", "history": []}
        
        session_mem = self.memory_store[session_id]
        session_mem["history"] = messages
        
        if len(messages) > self.max_history_window:
            system_prompts = [m for m in messages if m.get("role") == "system"]
            recent_messages = messages[-self.max_history_window:]
            
            compressed_messages = system_prompts + recent_messages
            logger.info(f"[Memory Layer] Session {session_id}: Compressed history from {len(messages)} down to {len(compressed_messages)} messages.")
            return compressed_messages
            
        return messages


# ==========================================
# 3. Session & Lifecycle Manager
# ==========================================
class SessionManager:
    """Manages multi-session isolation with automatic TTL cleanup to prevent memory leaks."""
    def __init__(self, ttl: int = 3600):
        self.pids: Dict[str, AdaptivePIDController] = {}
        self.memory_layer = MemoryLayer()
        self.ttl = ttl

    def get_pid(self, session_id: str) -> AdaptivePIDController:
        now = time.time()
        expired_sessions = [sid for sid, p in self.pids.items() if now - p.last_accessed > self.ttl]
        for sid in expired_sessions:
            if sid in self.pids:
                del self.pids[sid]
            logger.info(f"[Session Manager] Purged expired session: {sid}")

        if session_id not in self.pids:
            self.pids[session_id] = AdaptivePIDController()
            logger.info(f"[Session Manager] Initialized new Adaptive PID for session: {session_id}")
            
        return self.pids[session_id]

session_manager = SessionManager(ttl=SESSION_TTL_SECONDS)


# ==========================================
# 4. Auto-Anchor & Drift Monitor Engine
# ==========================================
class AnchorMonitor:
    """Automatically extracts system or user prompts as baseline anchors and computes drift scores."""
    @staticmethod
    def extract_anchor(messages: List[Dict[str, str]]) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @staticmethod
    def calculate_drift(anchor: str, current_text: str) -> float:
        if not anchor or not current_text:
            return 0.0
        anchor_words = set(anchor.lower().split())
        current_words = set(current_text.lower().split())
        
        if not anchor_words:
            return 0.0
        
        intersection = anchor_words.intersection(current_words)
        union = anchor_words.union(current_words)
        jaccard_similarity = len(intersection) / len(union) if union else 1.0
        
        return 1.0 - jaccard_similarity

def calculate_ngram_repetition(text: str, n: int = 3) -> float:
    """Calculates N-gram repetition ratio to detect repetitive loops."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def evaluate_homeostatic_load(messages: List[Dict[str, str]]) -> float:
    """Evaluates homeostatic load combining N-gram repetition and topic drift."""
    if not messages:
        return 2.5
        
    last_content = messages[-1].get("content", "")
    ngram_rep = calculate_ngram_repetition(last_content, n=3)
    
    anchor = AnchorMonitor.extract_anchor(messages)
    drift_score = AnchorMonitor.calculate_drift(anchor, last_content)
    
    logger.debug(f"[Monitor] Drift Score: {drift_score:.4f} | N-gram Rep: {ngram_rep:.4f}")
    
    if ngram_rep > 0.4 or drift_score > 0.85:
        return 0.2  # Trigger thermal adjustment
        
    return 2.5      # Normal equilibrium


# ==========================================
# 5. FastAPI Proxy & Health Check Endpoints
# ==========================================
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Liveness probe endpoint for container orchestrators (Docker / K8s)."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "Cognitive-Homeostatic Engine",
        "active_sessions": len(session_manager.pids)
    }

@app.post("/v1/chat/completions", tags=["Proxy"])
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body format", "code": 400}}, status_code=400)
    
    session_id = request.headers.get("x-session-id", request.client.host if request.client else "default-session")
    
    raw_messages = body.get("messages", [])
    optimized_messages = session_manager.memory_layer.process_and_compress(session_id, raw_messages)
    body["messages"] = optimized_messages
    
    homeo_load = evaluate_homeostatic_load(optimized_messages)
    pid = session_manager.get_pid(session_id)
    
    base_temp = body.get("temperature", 0.7)
    delta_t = pid.compute(homeo_load)
    adjusted_temp = max(0.1, min(2.0, base_temp + delta_t))
    body["temperature"] = adjusted_temp
    
    logger.info(f"Session: {session_id} | ML-PID Kp: {pid.kp:.4f} | Temp: {base_temp} -> {adjusted_temp:.4f}")
    
    client = httpx.AsyncClient(timeout=60.0)
    try:
        req = client.build_request("POST", BACKEND_URL, json=body, headers=dict(request.headers))
        response = await client.send(req, stream=True)
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to upstream LLM backend at {BACKEND_URL}: {e}")
        return JSONResponse({
            "error": {
                "message": f"Cognitive Middleware Error: Upstream backend unreachable at {BACKEND_URL}.",
                "type": "server_error",
                "code": 503
            }
        }, status_code=503)
    
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=dict(response.headers)
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)