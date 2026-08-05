# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Anti-Entropy Cognitive Middleware with Production Hardening (v2.3.0)
Author: Starsand
Jurisdiction: HKSAR (Hong Kong Special Administrative Region)
License: PolyForm Noncommercial License 1.0.0 (Non-Commercial Use Only)
"""

import os
import time
import asyncio
import logging
import httpx
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ==========================================
# 0. 系統日誌與環境配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CognitiveMiddleware")

app = FastAPI(
    title="Anti-Entropy Cognitive Middleware",
    description="Production-hardened cognitive homeostatic proxy with async locks, continuous metrics, and safe streaming.",
    version="2.3.0"
)

BACKEND_URL = os.getenv("LLM_BACKEND_URL", "http://localhost:8080/v1/chat/completions")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 64))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 4000))
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", 60.0))
UPSTREAM_RETRIES = int(os.getenv("UPSTREAM_RETRIES", 2))

# Hop-by-hop headers that must not be forwarded
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"
}

# Explicit whitelist of headers we allow to forward to upstream
FORWARD_HEADER_WHITELIST = {
    "accept",
    "accept-encoding",
    "content-type",
    "user-agent",
    "x-request-id",
    "x-session-id",
}

# ==========================================
# 1. 強健的機器學習自適應 PID 控制器
# ==========================================
class AdaptivePIDController:
    """
    具備積分防飽和（Anti-Windup）與線上增強學習適應機制的 PID 控制器。
    """
    def __init__(self, kp: float = 0.1, ki: float = 0.01, kd: float = 0.05, target_entropy: float = 2.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_entropy = target_entropy
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.success_streak = 0

    def compute(self, current_metric: float) -> float:
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-6

        error = self.target_entropy - current_metric
        
        # 積分防飽和夾具 (Anti-Windup Clamping)
        self.integral = max(-10.0, min(10.0, self.integral + (error * dt)))
        derivative = (error - self.last_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # 線上自適應機器學習調校
        if abs(error) < abs(self.last_error):
            self.success_streak += 1
            if self.success_streak > 3:
                self.kp = max(0.05, min(0.3, self.kp * 1.02))
                self.success_streak = 0
        else:
            self.kp = max(0.05, self.kp * 0.95)
            self.success_streak = 0

        self.last_error = error
        self.last_time = now
        return output


# ==========================================
# 2. 對話記憶與上下文壓縮層
# ==========================================
class MemoryLayer:
    def __init__(self, max_history_window: int = 12):
        self.max_history_window = max_history_window

    def process_and_compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # enforce limits and avoid returning references to caller's list
        msgs = list(messages or [])
        # trim individual message content length
        for m in msgs:
            if isinstance(m.get("content"), str) and len(m["content"]) > MAX_MESSAGE_LENGTH:
                m = m.copy()
                m["content"] = m["content"][:MAX_MESSAGE_LENGTH]
        if len(msgs) > self.max_history_window:
            system_prompts = [m for m in msgs if m.get("role") == "system"]
            recent_messages = msgs[-self.max_history_window:]
            # avoid duplicates between system_prompts and recent_messages
            combined = system_prompts + [m for m in recent_messages if m not in system_prompts]
            return combined
        return msgs


# ==========================================
# 3. 具備 Async Lock 的線程安全 Session 管理器
# ==========================================
class SessionEntry:
    def __init__(self):
        self.pid = AdaptivePIDController()
        self.lock = asyncio.Lock()
        self.last_accessed = time.time()

class SessionManager:
    """管理多用戶隔離實例，並透過 asyncio.Lock 確保高併發線程安全。"""
    def __init__(self, ttl: int = 3600):
        self.sessions: Dict[str, SessionEntry] = {}
        self.memory_layer = MemoryLayer()
        self.ttl = ttl
        self._sessions_lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str) -> SessionEntry:
        now = time.time()
        # 清理逾期 Session (best-effort)
        expired = [sid for sid, entry in list(self.sessions.items()) if now - entry.last_accessed > self.ttl]
        for sid in expired:
            self.sessions.pop(sid, None)

        # protect creation with a lock to avoid race conditions
        async with self._sessions_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionEntry()
                logger.info(f"[Session Manager] Initialized session: {session_id}")
            entry = self.sessions[session_id]
            entry.last_accessed = now
            return entry

session_manager = SessionManager(ttl=SESSION_TTL_SECONDS)


# ==========================================
# 4. 連續型穩態負載與漂移監控引擎
# ==========================================
class AnchorMonitor:
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
        return 1.0 - (len(intersection) / len(union) if union else 1.0)

def calculate_ngram_repetition(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))

def evaluate_continuous_homeostatic_load(messages: List[Dict[str, str]]) -> float:
    """計算連續型穩態負載指標，讓 PID 獲得平滑的誤差變化量。"""
    if not messages:
        return 2.5
    last_content = messages[-1].get("content", "")
    ngram_rep = calculate_ngram_repetition(last_content, n=3)
    anchor = AnchorMonitor.extract_anchor(messages)
    drift_score = AnchorMonitor.calculate_drift(anchor, last_content)
    
    # 透過加權組合計算連續性負載分數 (0.1 ~ 5.0)
    load = 2.5 - (ngram_rep * 1.5 + drift_score * 1.0)
    return max(0.1, min(5.0, load))


# ==========================================
# 5. FastAPI 安全代理與健康檢查端點
# ==========================================
@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {
        "status": "healthy",
        "version": "2.3.0",
        "active_sessions": len(session_manager.sessions)
    }


def _sanitize_and_limit_messages(raw_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs = list(raw_messages or [])
    if len(msgs) > MAX_MESSAGES:
        logger.warning(f"Trimming messages from {len(msgs)} to MAX_MESSAGES={MAX_MESSAGES}")
        msgs = msgs[-MAX_MESSAGES:]
    # truncate long content fields
    sanitized = []
    for m in msgs:
        mm = dict(m)
        if isinstance(mm.get("content"), str) and len(mm["content"]) > MAX_MESSAGE_LENGTH:
            mm["content"] = mm["content"][:MAX_MESSAGE_LENGTH]
        sanitized.append(mm)
    return sanitized


async def _send_with_retries(client: httpx.AsyncClient, req: httpx.Request) -> httpx.Response:
    last_exc = None
    for attempt in range(1, UPSTREAM_RETRIES + 2):
        try:
            response = await client.send(req, stream=True)
            # treat 5xx as retriable
            if response.status_code >= 500 and attempt <= UPSTREAM_RETRIES:
                await response.aclose()
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"Upstream 5xx, retrying in {backoff}s (attempt {attempt})")
                await asyncio.sleep(backoff)
                continue
            return response
        except (httpx.RequestError, httpx.TransportError) as e:
            last_exc = e
            backoff = 0.5 * (2 ** (attempt - 1))
            logger.warning(f"Upstream request error: {e}; retrying in {backoff}s (attempt {attempt})")
            await asyncio.sleep(backoff)
            continue
    # if we get here, all retries exhausted
    raise last_exc if last_exc is not None else RuntimeError("Upstream retries exhausted")


@app.post("/v1/chat/completions", tags=["Proxy"])
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body format", "code": 400}}, status_code=400)
    
    session_id = request.headers.get("x-session-id", request.client.host if request.client else "default-session")
    
    # 取得安全隔離的 Session 實例與鎖
    session_entry = await session_manager.get_or_create_session(session_id)

    # sanitize and limit messages early
    raw_messages = body.get("messages", [])
    raw_messages = _sanitize_and_limit_messages(raw_messages)
    optimized_messages = session_manager.memory_layer.process_and_compress(raw_messages)
    body["messages"] = optimized_messages

    # compute homeostatic load and adjust temperature under session lock
    async with session_entry.lock:
        homeo_load = evaluate_continuous_homeostatic_load(optimized_messages)
        delta_t = session_entry.pid.compute(homeo_load)
        base_temp = body.get("temperature", 0.7)
        adjusted_temp = max(0.1, min(2.0, base_temp + delta_t))
        body["temperature"] = adjusted_temp
        session_entry.last_accessed = time.time()

    logger.info(f"Session: {session_id} | Load: {homeo_load:.2f} | Temp: {base_temp} -> {adjusted_temp:.4f}")

    # 過濾並白名單傳入標頭
    forward_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in FORWARD_HEADER_WHITELIST and kl not in HOP_BY_HOP_HEADERS:
            forward_headers[k] = v

    # build request and send with async client using context manager and retry
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        req = client.build_request("POST", BACKEND_URL, json=body, headers=forward_headers)
        try:
            response = await _send_with_retries(client, req)
        except Exception as e:
            logger.error(f"Upstream unreachable at {BACKEND_URL}: {e}")
            return JSONResponse({
                "error": {
                    "message": f"Middleware Error: Upstream backend unreachable.",
                    "type": "server_error",
                    "code": 503
                }
            }, status_code=503)

        # 安全串流回傳並確保資源正確釋放
        async def stream_and_close():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        resp_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}

        return StreamingResponse(
            stream_and_close(),
            status_code=response.status_code,
            headers=resp_headers
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
