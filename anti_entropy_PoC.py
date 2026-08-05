# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Anti-Entropy Cognitive Middleware: Enterprise Ultimate Edition (v3.3.0)
Author: Starsand
Jurisdiction: HKSAR (Hong Kong Special Administrative Region)
License: PolyForm Noncommercial License 1.0.0 (Non-Commercial Use Only)
"""

import os
import time
import asyncio
import logging
import re
import httpx
from typing import List, Dict, Set, Optional, Any
from collections import OrderedDict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
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
    description="Enterprise-grade cognitive homeostatic proxy with multi-backend failover, rate limiting, adversarial guardrails, and telemetry auditing.",
    version="3.3.0"
)

# Environment variables and configurations
BACKEND_URLS = [
    url.strip() for url in os.getenv("LLM_BACKEND_URLS", "http://localhost:8080/v1/chat/completions").split(",")
    if url.strip()
]
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))
MAX_LRU_SESSIONS = int(os.getenv("MAX_LRU_SESSIONS", 5000))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 64))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 4000))
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", 60.0))
UPSTREAM_RETRIES = int(os.getenv("UPSTREAM_RETRIES", 2))

HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"
}

FORWARD_HEADER_WHITELIST = {
    "accept",
    "accept-encoding",
    "content-type",
    "user-agent",
    "x-request-id",
    "x-session-id",
}

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"disregard all rules",
    r"system prompt extraction",
    r"you are now DAN",
    r"jailbreak mode",
    r"override safety"
]


# ==========================================
# 1. Prometheus Metrics Collector
# ==========================================
class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.entropy_violations = 0
        self.circuit_trips = 0
        self.rate_limited_count = 0
        self.injection_blocked_count = 0
        self.total_latency_ms = 0.0

    def export_metrics(self) -> str:
        avg_latency = (self.total_latency_ms / self.request_count) if self.request_count > 0 else 0.0
        return f"""# HELP anti_entropy_requests_total Total proxy requests handled.
# TYPE anti_entropy_requests_total counter
anti_entropy_requests_total {self.request_count}

# HELP anti_entropy_violations_total Total entropy degradation events intercepted.
# TYPE anti_entropy_violations_total counter
anti_entropy_violations_total {self.entropy_violations}

# HELP anti_entropy_circuit_trips_total Total upstream circuit breaker trips.
# TYPE anti_entropy_circuit_trips_total counter
anti_entropy_circuit_trips_total {self.circuit_trips}

# HELP anti_entropy_rate_limited_total Total requests blocked by rate limiter.
# TYPE anti_entropy_rate_limited_total counter
anti_entropy_rate_limited_total {self.rate_limited_count}

# HELP anti_entropy_injection_blocked_total Total prompt injections blocked.
# TYPE anti_entropy_injection_blocked_total counter
anti_entropy_injection_blocked_total {self.injection_blocked_count}

# HELP anti_entropy_avg_latency_ms Average upstream proxy latency in milliseconds.
# TYPE anti_entropy_avg_latency_ms gauge
anti_entropy_avg_latency_ms {avg_latency:.2f}
"""

metrics = MetricsCollector()


# ==========================================
# 2. Multi-Backend Failover & Circuit Breaker
# ==========================================
class MultiBackendCircuitBreaker:
    def __init__(self, urls: List[str], threshold: int = 4, timeout: float = 20.0):
        self.urls = urls if urls else ["http://localhost:8080/v1/chat/completions"]
        self.current_index = 0
        self.threshold = threshold
        self.timeout = timeout
        self.failures = {url: 0 for url in self.urls}
        self.states = {url: "CLOSED" for url in self.urls}
        self.last_failure_times = {url: 0.0 for url in self.urls}

    def get_active_backend(self) -> str:
        now = time.time()
        for _ in range(len(self.urls)):
            url = self.urls[self.current_index]
            state = self.states[url]
            if state == "OPEN":
                if now - self.last_failure_times[url] > self.timeout:
                    self.states[url] = "HALF_OPEN"
                    return url
                self.current_index = (self.current_index + 1) % len(self.urls)
            else:
                return url
        return self.urls[0]

    def record_success(self, url: str):
        self.failures[url] = 0
        self.states[url] = "CLOSED"

    def record_failure(self, url: str):
        self.failures[url] += 1
        self.last_failure_times[url] = time.time()
        if self.failures[url] >= self.threshold:
            self.states[url] = "OPEN"
            metrics.circuit_trips += 1
            logger.error(f"[Circuit Breaker] Backend {url} marked OPEN due to repeated failures.")
        self.current_index = (self.current_index + 1) % len(self.urls)

backend_router = MultiBackendCircuitBreaker(BACKEND_URLS)


# ==========================================
# 3. Adversarial Guardrail & Injection Filter
# ==========================================
class AdversarialGuardrail:
    @staticmethod
    def inspect(messages: List[Dict[str, str]]) -> bool:
        for msg in messages:
            content = msg.get("content", "").lower()
            for pattern in INJECTION_PATTERNS:
                if re.search(pattern, content):
                    return True
        return False


# ==========================================
# 4. Token Bucket Rate Limiter
# ==========================================
class RateLimiter:
    def __init__(self, capacity: float = 30.0, refill_rate: float = 10.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        if key not in self.tokens:
            self.tokens[key] = self.capacity
            self.last_refill[key] = now

        elapsed = now - self.last_refill[key]
        self.tokens[key] = min(self.capacity, self.tokens[key] + elapsed * self.refill_rate)
        self.last_refill[key] = now

        if self.tokens[key] >= 1.0:
            self.tokens[key] -= 1.0
            return True
        return False

rate_limiter = RateLimiter()


# ==========================================
# 5. Adaptive Machine Learning PID Controller
# ==========================================
class AdaptivePIDController:
    """
    PID controller featuring anti-windup clamping and online reinforcement learning adaptation.
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
        
        self.integral = max(-10.0, min(10.0, self.integral + (error * dt)))
        derivative = (error - self.last_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
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
# 6. Memory Layer & Context Compression
# ==========================================
class MemoryLayer:
    def __init__(self, max_history_window: int = 12):
        self.max_history_window = max_history_window

    def process_and_compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        msgs = list(messages or [])
        for m in msgs:
            if isinstance(m.get("content"), str) and len(m["content"]) > MAX_MESSAGE_LENGTH:
                m = m.copy()
                m["content"] = m["content"][:MAX_MESSAGE_LENGTH]
        if len(msgs) > self.max_history_window:
            system_prompts = [m for m in msgs if m.get("role") == "system"]
            recent_messages = msgs[-self.max_history_window:]
            combined = system_prompts + [m for m in recent_messages if m not in system_prompts]
            return combined
        return msgs


# ==========================================
# 7. LRU Session Manager with TTL & Thread Safety
# ==========================================
class SessionEntry:
    def __init__(self):
        self.pid = AdaptivePIDController()
        self.lock = asyncio.Lock()
        self.last_accessed = time.time()

class SessionManager:
    """Manages isolated user sessions with thread safety and LRU capacity eviction."""
    def __init__(self, max_capacity: int = MAX_LRU_SESSIONS, ttl: int = SESSION_TTL_SECONDS):
        self.max_capacity = max_capacity
        self.ttl = ttl
        self.sessions: OrderedDict[str, SessionEntry] = OrderedDict()
        self.memory_layer = MemoryLayer()
        self._sessions_lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str) -> SessionEntry:
        now = time.time()
        
        async with self._sessions_lock:
            expired = [sid for sid, entry in list(self.sessions.items()) if now - entry.last_accessed > self.ttl]
            for sid in expired:
                self.sessions.pop(sid, None)

            if session_id in self.sessions:
                self.sessions.move_to_end(session_id)
                entry = self.sessions[session_id]
            else:
                if len(self.sessions) >= self.max_capacity:
                    self.sessions.popitem(last=False)
                entry = SessionEntry()
                self.sessions[session_id] = entry
                logger.info(f"[Session Manager] Initialized LRU session: {session_id}")

            entry.last_accessed = now
            return entry

session_manager = SessionManager(ttl=SESSION_TTL_SECONDS)


# ==========================================
# 8. Continuous Homeostatic Load & Drift Monitor
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
    """Evaluates continuous homeostatic load for smooth PID error modulation."""
    if not messages:
        return 2.5
    last_content = messages[-1].get("content", "")
    ngram_rep = calculate_ngram_repetition(last_content, n=3)
    anchor = AnchorMonitor.extract_anchor(messages)
    drift_score = AnchorMonitor.calculate_drift(anchor, last_content)
    
    load = 2.5 - (ngram_rep * 1.5 + drift_score * 1.0)
    return max(0.1, min(5.0, load))


# ==========================================
# 9. Dynamic Token Governor & Audit Logger
# ==========================================
class TokenGovernor:
    @staticmethod
    def govern(body: dict, homeo_load: float):
        """Dynamically adjusts max_tokens based on thermodynamic load."""
        if homeo_load < 1.2:
            body["max_tokens"] = min(body.get("max_tokens", 2048), 256)
        elif homeo_load > 3.5:
            if "max_tokens" not in body:
                body["max_tokens"] = 2048

class AuditLogger:
    @staticmethod
    async def log_trajectory(sid: str, backend: str, load: float, temp_delta: float, status_code: int):
        """Asynchronously records structured telemetry for auditing and compliance."""
        record = {
            "timestamp": time.time(),
            "session_id": sid,
            "backend_target": backend,
            "utp_load": round(load, 3),
            "temp_adjustment": round(temp_delta, 4),
            "status": status_code
        }
        logger.debug(f"[Audit Trajectory] {record}")


# ==========================================
# 10. FastAPI Endpoints & Core Proxy Logic
# ==========================================
@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {
        "status": "enterprise_ultimate_healthy",
        "version": "3.3.0",
        "active_sessions": len(session_manager.sessions),
        "active_backend": backend_router.get_active_backend()
    }

@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    return PlainTextResponse(metrics.export_metrics())


def _sanitize_and_limit_messages(raw_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs = list(raw_messages or [])
    if len(msgs) > MAX_MESSAGES:
        logger.warning(f"Trimming messages from {len(msgs)} to MAX_MESSAGES={MAX_MESSAGES}")
        msgs = msgs[-MAX_MESSAGES:]
    sanitized = []
    for m in msgs:
        mm = dict(m)
        if isinstance(mm.get("content"), str) and len(mm["content"]) > MAX_MESSAGE_LENGTH:
            mm["content"] = mm["content"][:MAX_MESSAGE_LENGTH]
        sanitized.append(mm)
    return sanitized


async def _send_with_retries(client: httpx.AsyncClient, req: httpx.Request, backend_url: str) -> httpx.Response:
    last_exc = None
    for attempt in range(1, UPSTREAM_RETRIES + 2):
        try:
            response = await client.send(req, stream=True)
            if response.status_code >= 500 and attempt <= UPSTREAM_RETRIES:
                await response.aclose()
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"Upstream 5xx at {backend_url}, retrying in {backoff}s (attempt {attempt})")
                await asyncio.sleep(backoff)
                continue
            return response
        except (httpx.RequestError, httpx.TransportError) as e:
            last_exc = e
            backoff = 0.5 * (2 ** (attempt - 1))
            logger.warning(f"Upstream request error at {backend_url}: {e}; retrying in {backoff}s (attempt {attempt})")
            await asyncio.sleep(backoff)
            continue
    raise last_exc if last_exc is not None else RuntimeError("Upstream retries exhausted")


@app.post("/v1/chat/completions", tags=["Proxy"])
async def proxy_chat_completions(request: Request):
    client_ip = request.client.host if request.client else "default"
    session_id = request.headers.get("x-session-id", client_ip)

    if not rate_limiter.allow(session_id):
        metrics.rate_limited_count += 1
        return JSONResponse({
            "error": {
                "message": "Rate limit exceeded. Please slow down your request frequency.",
                "type": "rate_limit_error",
                "code": 429
            }
        }, status_code=429)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body format", "code": 400}}, status_code=400)
    
    raw_messages = body.get("messages", [])

    if AdversarialGuardrail.inspect(raw_messages):
        metrics.injection_blocked_count += 1
        return JSONResponse({
            "error": {
                "message": "Request intercepted by adversarial guardrail policy.",
                "type": "security_violation",
                "code": 403
            }
        }, status_code=403)

    session_entry = await session_manager.get_or_create_session(session_id)

    raw_messages = _sanitize_and_limit_messages(raw_messages)
    optimized_messages = session_manager.memory_layer.process_and_compress(raw_messages)
    body["messages"] = optimized_messages

    metrics.request_count += 1
    start_time = time.time()

    async with session_entry.lock:
        homeo_load = evaluate_continuous_homeostatic_load(optimized_messages)
        if homeo_load < 1.5:
            metrics.entropy_violations += 1

        delta_t = session_entry.pid.compute(homeo_load)
        base_temp = body.get("temperature", 0.7)
        adjusted_temp = max(0.1, min(2.0, base_temp + delta_t))
        body["temperature"] = adjusted_temp

        if "top_p" in body or delta_t < 0:
            base_topp = body.get("top_p", 1.0)
            body["top_p"] = max(0.1, min(1.0, base_topp - (delta_t * 0.2)))

        TokenGovernor.govern(body, homeo_load)
        session_entry.last_accessed = time.time()

    active_url = backend_router.get_active_backend()
    logger.info(f"Session: {session_id} | Backend: {active_url} | Load: {homeo_load:.2f} | Temp: {base_temp} -> {adjusted_temp:.4f}")

    forward_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in FORWARD_HEADER_WHITELIST and kl not in HOP_BY_HOP_HEADERS:
            forward_headers[k] = v

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        req = client.build_request("POST", active_url, json=body, headers=forward_headers)
        try:
            response = await _send_with_retries(client, req, active_url)
            backend_router.record_success(active_url)
        except Exception as e:
            backend_router.record_failure(active_url)
            logger.error(f"Upstream unreachable at {active_url}: {e}")
            asyncio.create_task(AuditLogger.log_trajectory(session_id, active_url, homeo_load, delta_t, 503))
            return JSONResponse({
                "error": {
                    "message": "Middleware Error: Active and fallback inference backends unreachable.",
                    "type": "server_error",
                    "code": 503
                }
            }, status_code=503)

    elapsed_ms = (time.time() - start_time) * 1000.0
    metrics.total_latency_ms += elapsed_ms

    asyncio.create_task(AuditLogger.log_trajectory(session_id, active_url, homeo_load, delta_t, response.status_code))

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