# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Anti-Entropy Cognitive Middleware: Enterprise Production Edition (v4.0.2)
Author: Starsand
Jurisdiction: HKSAR (Hong Kong Special Administrative Region)
License: PolyForm Noncommercial License 1.0.0 (Non-Commercial Use Only)

Multi-Pod Scaling Architecture Note (Redis Adapter Skeleton):
To scale horizontally across multiple pods, replace process-local RateLimiter & SessionManager 
with a Redis cluster backing store:
    class RedisSessionManager:
        async def get_or_create_session(self, session_id: str):
            # Redis hash or JSON module lookup for session state & sliding window
            pass
"""

import os
import time
import asyncio
import logging
import re
import hashlib
import json
import random
from typing import List, Dict, Set, Optional, Any
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import httpx

# Optional: prometheus_client for standard metrics
try:
    from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS_LIB = True
except ImportError:
    HAS_PROMETHEUS_LIB = False

# Optional: tiktoken import for precise token counting
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    ENCODER = None

# ==========================================
# 0. Global Environment Variables & Configs
# ==========================================
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"
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
MAX_MODEL_CONTEXT = int(os.getenv("MAX_MODEL_CONTEXT", 8192))
REANCHOR_COOLDOWN_SECONDS = float(os.getenv("REANCHOR_COOLDOWN", 45.0))
MAX_REANCHORS_PER_SESSION = int(os.getenv("MAX_REANCHORS_PER_SESSION", 3))

# Connection Pool Settings
HTTP_MAX_KEEPALIVE = int(os.getenv("HTTP_MAX_KEEPALIVE", 50))
HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", 200))

# System Logging Setup
logging.basicConfig(
    level=logging.DEBUG if DEBUG_LOGGING else logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CognitiveMiddleware")

# Global HTTP Client Instance
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    limits = httpx.Limits(
        max_keepalive_connections=HTTP_MAX_KEEPALIVE,
        max_connections=HTTP_MAX_CONNECTIONS
    )
    http_client = httpx.AsyncClient(
        timeout=UPSTREAM_TIMEOUT,
        limits=limits,
        http2=True
    )
    logger.info(f"[Lifespan] Connection pool ready (KeepAlive: {HTTP_MAX_KEEPALIVE}, MaxConn: {HTTP_MAX_CONNECTIONS})")
    yield
    await http_client.aclose()
    logger.info("[Lifespan] Global HTTP connection pool closed.")

app = FastAPI(
    title="Anti-Entropy Cognitive Middleware",
    description="Enterprise-grade cognitive homeostatic proxy with robust metric name mapping, expanded K8s health checks, and session reset support.",
    version="4.0.2",
    lifespan=lifespan
)

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
# 1. Prometheus Metrics Collector (Fixed Name Mapping)
# ==========================================
class PrometheusMetrics:
    def __init__(self):
        self.metric_mapping = {
            "requests": "requests",
            "violations": "entropy_violations",
            "circuit_trips": "circuit_trips",
            "rate_limited": "rate_limited",
            "injections": "injections",
            "loops": "loops",
            "tokens_sent": "tokens_sent",
            "tokens_saved": "tokens_saved",
            "reanchors": "reanchors"
        }
        if HAS_PROMETHEUS_LIB:
            self.requests = Counter("anti_entropy_requests_total", "Total proxy requests handled.")
            self.entropy_violations = Counter("anti_entropy_violations_total", "Total entropy degradation events.")
            self.circuit_trips = Counter("anti_entropy_circuit_trips_total", "Total circuit breaker trips.")
            self.rate_limited = Counter("anti_entropy_rate_limited_total", "Total rate limited requests.")
            self.injections = Counter("anti_entropy_injection_blocked_total", "Total prompt injections blocked.")
            self.loops = Counter("anti_entropy_loop_detected_total", "Total loop generations detected.")
            self.tokens_sent = Counter("anti_entropy_tokens_sent_total", "Cumulative tokens sent.")
            self.tokens_saved = Counter("anti_entropy_tokens_saved_total", "Cumulative tokens saved.")
            self.reanchors = Counter("anti_entropy_reanchors_total", "Total re-anchoring events.")
            self.latency_gauge = Gauge("anti_entropy_avg_latency_ms", "Latest proxy request latency in ms.")
        else:
            self._lock = asyncio.Lock()
            self._counters = {
                "requests": 0, "violations": 0, "circuit_trips": 0,
                "rate_limited": 0, "injections": 0, "loops": 0,
                "tokens_sent": 0, "tokens_saved": 0, "reanchors": 0
            }
            self.latest_latency = 0.0

    async def increment(self, metric_name: str, amount: float = 1.0):
        target_attr = self.metric_mapping.get(metric_name, metric_name)
        if HAS_PROMETHEUS_LIB:
            gauge_or_counter = getattr(self, target_attr, None)
            if gauge_or_counter and hasattr(gauge_or_counter, "inc"):
                gauge_or_counter.inc(amount)
        else:
            async with self._lock:
                if metric_name in self._counters:
                    self._counters[metric_name] += amount

    async def update_latency(self, latency_ms: float):
        if HAS_PROMETHEUS_LIB:
            self.latency_gauge.set(latency_ms)
        else:
            async with self._lock:
                self.latest_latency = latency_ms

    async def render(self) -> tuple[bytes, str]:
        if HAS_PROMETHEUS_LIB:
            return generate_latest(), CONTENT_TYPE_LATEST
        else:
            async with self._lock:
                c = self._counters
                output = f"""# HELP anti_entropy_requests_total Total proxy requests handled.
# TYPE anti_entropy_requests_total counter
anti_entropy_requests_total {c['requests']}
# HELP anti_entropy_violations_total Total entropy degradation events.
# TYPE anti_entropy_violations_total counter
anti_entropy_violations_total {c['violations']}
# HELP anti_entropy_circuit_trips_total Total circuit breaker trips.
# TYPE anti_entropy_circuit_trips_total counter
anti_entropy_circuit_trips_total {c['circuit_trips']}
# HELP anti_entropy_rate_limited_total Total rate limited requests.
# TYPE anti_entropy_rate_limited_total counter
anti_entropy_rate_limited_total {c['rate_limited']}
# HELP anti_entropy_injection_blocked_total Total prompt injections blocked.
# TYPE anti_entropy_injection_blocked_total counter
anti_entropy_injection_blocked_total {c['injections']}
# HELP anti_entropy_loop_detected_total Total loop generations detected.
# TYPE anti_entropy_loop_detected_total counter
anti_entropy_loop_detected_total {c['loops']}
# HELP anti_entropy_tokens_sent_total Cumulative tokens sent.
# TYPE anti_entropy_tokens_sent_total counter
anti_entropy_tokens_sent_total {c['tokens_sent']}
# HELP anti_entropy_tokens_saved_total Cumulative tokens saved.
# TYPE anti_entropy_tokens_saved_total counter
anti_entropy_tokens_saved_total {c['tokens_saved']}
# HELP anti_entropy_reanchors_total Total re-anchoring events.
# TYPE anti_entropy_reanchors_total counter
anti_entropy_reanchors_total {c['reanchors']}
# HELP anti_entropy_avg_latency_ms Latest proxy request latency in ms.
# TYPE anti_entropy_avg_latency_ms gauge
anti_entropy_avg_latency_ms {self.latest_latency:.2f}
"""
            return output.encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8"

metrics = PrometheusMetrics()


# ==========================================
# 2. Circuit Breaker & Failover Router
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

    async def record_failure_async(self, url: str):
        self.failures[url] += 1
        self.last_failure_times[url] = time.time()
        if self.failures[url] >= self.threshold:
            self.states[url] = "OPEN"
            await metrics.increment("circuit_trips")
            logger.error(f"[Circuit Breaker] Backend {url} set to OPEN due to repeated failures.")
        self.current_index = (self.current_index + 1) % len(self.urls)

backend_router = MultiBackendCircuitBreaker(BACKEND_URLS)


# ==========================================
# 3. Guardrails & Token Utility
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

def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    serialized = "".join([f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n" for m in messages])
    if not serialized:
        return 0
    if ENCODER is not None:
        try:
            return len(ENCODER.encode(serialized))
        except Exception:
            pass
    return max(1, int(len(serialized) / 2.5))


# ==========================================
# 4. Rate Limiter (Process-Local Bucket)
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
# 5. Adaptive PID Controller
# ==========================================
class AdaptivePIDController:
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
# 6. Memory Layer & Loop Guard
# ==========================================
class MemoryLayer:
    def __init__(self, max_history_window: int = 12):
        self.max_history_window = max_history_window

    def sanitize_and_validate_roles(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        valid_roles = {"system", "user", "assistant"}
        sanitized = []
        for m in messages:
            role = m.get("role", "user")
            if role not in valid_roles:
                role = "user"
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > MAX_MESSAGE_LENGTH:
                content = content[:MAX_MESSAGE_LENGTH]
            sanitized.append({"role": role, "content": content})
        return sanitized

    async def process_and_compress(self, messages: List[Dict[str, str]], session_system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        msgs = self.sanitize_and_validate_roles(messages)
        tokens_before = count_messages_tokens(msgs)

        system_prompts = [m for m in msgs if m.get("role") == "system"]
        canonical_prompt = session_system_prompt or (system_prompts[0]["content"] if system_prompts else "You are a precise, reliable AI assistant.")
        
        non_system_msgs = [m for m in msgs if m.get("role") != "system"]
        if len(non_system_msgs) > self.max_history_window:
            non_system_msgs = non_system_msgs[-self.max_history_window:]
            
        final_msgs = [{"role": "system", "content": canonical_prompt}] + non_system_msgs
        tokens_after = count_messages_tokens(final_msgs)

        saved = tokens_before - tokens_after
        if saved > 0:
            await metrics.increment("tokens_saved", saved)

        return final_msgs


# ==========================================
# 7. Session Manager (LRU + TTL)
# ==========================================
class SessionEntry:
    def __init__(self):
        self.pid = AdaptivePIDController()
        self.lock = asyncio.Lock()
        self.last_accessed = time.time()
        self.system_prompt: Optional[str] = None
        self.last_reanchor_time = 0.0
        self.reanchor_count = 0
        self.response_hashes: deque = deque(maxlen=5)

    async def check_loop_async(self, assistant_text: str) -> bool:
        async with self.lock:
            if not assistant_text or len(assistant_text.strip()) < 10:
                return False
            h = hashlib.sha256(assistant_text.strip().encode("utf-8")).hexdigest()
            if h in self.response_hashes:
                return True
            self.response_hashes.append(h)
            return False

class SessionManager:
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
# 8. Homeostatic Load Monitor
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
    if not messages:
        return 2.5
    last_content = messages[-1].get("content", "")
    ngram_rep = calculate_ngram_repetition(last_content, n=3)
    anchor = AnchorMonitor.extract_anchor(messages)
    drift_score = AnchorMonitor.calculate_drift(anchor, last_content)
    
    load = 2.5 - (ngram_rep * 1.5 + drift_score * 1.0)
    return max(0.1, min(5.0, load))


# ==========================================
# 9. Token Governor & Audit Logger
# ==========================================
class TokenGovernor:
    @staticmethod
    def govern(body: dict, homeo_load: float, prompt_tokens: int) -> Optional[JSONResponse]:
        safety_margin = 64
        remaining_budget = MAX_MODEL_CONTEXT - prompt_tokens - safety_margin
        
        if remaining_budget <= 0:
            return JSONResponse({
                "error": {
                    "message": "Context window exhausted. Prompt size exceeds total available context.",
                    "hint": "Please call POST /v1/session/reset to clear history, trim old messages, or execute context summarization.",
                    "type": "context_exhausted_error",
                    "code": 400
                }
            }, status_code=400)

        current_max = body.get("max_tokens", 2048)
        body["max_tokens"] = min(current_max, remaining_budget)

        if homeo_load < 1.2:
            body["max_tokens"] = min(body["max_tokens"], 256)
        elif homeo_load > 3.5:
            if "max_tokens" not in body:
                body["max_tokens"] = min(2048, remaining_budget)
        return None

class AuditLogger:
    @staticmethod
    async def log_trajectory(sid: str, backend: str, load: float, temp_delta: float, status_code: int):
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
# 10. K8s Health, Readiness & Metrics Endpoints
# ==========================================
@app.get("/health", tags=["Monitoring"])
@app.get("/live", tags=["Monitoring"])
async def liveness_check():
    return {"status": "alive", "version": "4.0.2"}

@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    if http_client is None:
        return JSONResponse({"status": "not_ready", "reason": "http_client_uninitialized"}, status_code=503)
    
    # Check if all backend circuit breakers are tripped (OPEN)
    all_open = all(state == "OPEN" for state in backend_router.states.values())
    if all_open and backend_router.urls:
        return JSONResponse({
            "status": "not_ready", 
            "reason": "all_backends_circuit_open"
        }, status_code=503)

    active_backend = backend_router.get_active_backend()
    return {
        "status": "ready",
        "active_backend": active_backend,
        "active_sessions": len(session_manager.sessions)
    }

@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    content_bytes, content_type = await metrics.render()
    return PlainTextResponse(content_bytes.decode("utf-8"), media_type=content_type)


# ==========================================
# 11. Session Management Endpoints (UX)
# ==========================================
@app.post("/v1/session/reset", tags=["Session"])
async def reset_session(request: Request):
    client_ip = request.client.host if request.client else "default"
    session_id = request.headers.get("x-session-id", client_ip)
    
    async with session_manager._sessions_lock:
        if session_id in session_manager.sessions:
            session_manager.sessions.pop(session_id, None)
            
    logger.info(f"[Session Reset] Session history manually cleared for session_id: {session_id}")
    return {
        "status": "success",
        "message": f"Session {session_id} history has been successfully reset.",
        "session_id": session_id
    }


# Retries with Jittered Exponential Backoff
async def _send_with_retries(req: httpx.Request, backend_url: str) -> httpx.Response:
    global http_client
    if http_client is None:
        raise RuntimeError("HTTP connection pool not initialized.")

    last_exc = None
    for attempt in range(1, UPSTREAM_RETRIES + 2):
        try:
            response = await http_client.send(req, stream=True)
            if response.status_code >= 500 and attempt <= UPSTREAM_RETRIES:
                await response.aclose()
                base_backoff = 0.5 * (2 ** (attempt - 1))
                jitter = random.uniform(0.05, 0.25)
                total_backoff = base_backoff + jitter
                logger.warning(f"Upstream 5xx at {backend_url}, retrying in {total_backoff:.2f}s (attempt {attempt})")
                await asyncio.sleep(total_backoff)
                continue
            return response
        except (httpx.RequestError, httpx.TransportError) as e:
            last_exc = e
            base_backoff = 0.5 * (2 ** (attempt - 1))
            jitter = random.uniform(0.05, 0.25)
            total_backoff = base_backoff + jitter
            logger.warning(f"Upstream error at {backend_url}: {e}; retrying in {total_backoff:.2f}s (attempt {attempt})")
            await asyncio.sleep(total_backoff)
            continue
    raise last_exc if last_exc is not None else RuntimeError("Upstream retries exhausted")


# ==========================================
# 12. Core Proxy Handler
# ==========================================
@app.post("/v1/chat/completions", tags=["Proxy"])
async def proxy_chat_completions(request: Request):
    await metrics.increment("requests")
    client_ip = request.client.host if request.client else "default"
    session_id = request.headers.get("x-session-id", client_ip)

    if not rate_limiter.allow(session_id):
        await metrics.increment("rate_limited")
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
        await metrics.increment("injections")
        return JSONResponse({
            "error": {
                "message": "Request intercepted by adversarial guardrail policy.",
                "type": "security_violation",
                "code": 403
            }
        }, status_code=403)

    session_entry = await session_manager.get_or_create_session(session_id)

    async with session_entry.lock:
        if not session_entry.system_prompt:
            for m in raw_messages:
                if m.get("role") == "system":
                    session_entry.system_prompt = m.get("content")
                    break

        optimized_messages = await session_manager.memory_layer.process_and_compress(raw_messages, session_entry.system_prompt)

        homeo_load = evaluate_continuous_homeostatic_load(optimized_messages)
        if homeo_load < 1.5:
            await metrics.increment("violations")

        now = time.time()
        if (
            homeo_load < 1.8 
            and session_entry.reanchor_count < MAX_REANCHORS_PER_SESSION 
            and (now - session_entry.last_reanchor_time > REANCHOR_COOLDOWN_SECONDS)
        ):
            reanchor_msg = {
                "role": "system",
                "content": "System Reminder: Maintain strict task alignment, logical rigor, and avoid circular or repetitive phrasing."
            }
            optimized_messages.insert(1, reanchor_msg)
            session_entry.last_reanchor_time = now
            session_entry.reanchor_count += 1
            await metrics.increment("reanchors")
            logger.info(f"[Re-anchor] Injected system prompt ({session_entry.reanchor_count}/{MAX_REANCHORS_PER_SESSION}) for session {session_id} (Load: {homeo_load:.2f})")
        elif session_entry.reanchor_count >= MAX_REANCHORS_PER_SESSION and homeo_load < 1.8:
            logger.warning(f"[Re-anchor Cap Reached] Session {session_id} hit reanchor limit ({MAX_REANCHORS_PER_SESSION}). Skipping.")

        body["messages"] = optimized_messages

        prompt_tokens = count_messages_tokens(optimized_messages)
        await metrics.increment("tokens_sent", prompt_tokens)

        governor_error = TokenGovernor.govern(body, homeo_load, prompt_tokens)
        if governor_error is not None:
            await metrics.increment("rate_limited")
            return governor_error

        delta_t = session_entry.pid.compute(homeo_load)
        base_temp = body.get("temperature", 0.7)
        adjusted_temp = max(0.1, min(2.0, base_temp + delta_t))
        body["temperature"] = adjusted_temp

        if "top_p" in body or delta_t < 0:
            base_topp = body.get("top_p", 1.0)
            body["top_p"] = max(0.1, min(1.0, base_topp - (delta_t * 0.2)))

        session_entry.last_accessed = now

    start_time = time.time()
    active_url = backend_router.get_active_backend()
    logger.info(f"Session: {session_id} | Backend: {active_url} | Load: {homeo_load:.2f} | Tokens: {prompt_tokens} | Temp: {base_temp} -> {adjusted_temp:.4f}")

    forward_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in FORWARD_HEADER_WHITELIST and kl not in HOP_BY_HOP_HEADERS:
            forward_headers[k] = v

    if http_client is None:
        return JSONResponse({"error": {"message": "Middleware not ready.", "code": 503}}, status_code=503)

    req = http_client.build_request("POST", active_url, json=body, headers=forward_headers)
    try:
        response = await _send_with_retries(req, active_url)
        backend_router.record_success(active_url)
    except Exception as e:
        await backend_router.record_failure_async(active_url)
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
    await metrics.update_latency(elapsed_ms)
    asyncio.create_task(AuditLogger.log_trajectory(session_id, active_url, homeo_load, delta_t, response.status_code))

    content_type = response.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            resp_body = await response.aread()
            await response.aclose()
            text_body = resp_body.decode("utf-8", errors="ignore")
            data = json.loads(text_body)
            
            assistant_text = ""
            choices = data.get("choices", [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    if "message" in first_choice and isinstance(first_choice["message"], dict):
                        assistant_text = first_choice["message"].get("content", "")
                    elif "text" in first_choice:
                        assistant_text = first_choice.get("text", "")
            
            if assistant_text:
                is_dup = await session_entry.check_loop_async(assistant_text)
                if is_dup or calculate_ngram_repetition(assistant_text, n=3) > 0.6:
                    await metrics.increment("loops")
                    logger.warning(f"[Loop Detector] Detected repetition in non-streaming response for session {session_id}")
            
            resp_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}
            return JSONResponse(data, status_code=response.status_code, headers=resp_headers)
        except Exception as ex:
            logger.error(f"[JSON Parse Error] {ex}")
            return JSONResponse({
                "error": {
                    "message": "Middleware Error: Failed to parse upstream JSON payload structure.",
                    "type": "upstream_parse_error",
                    "code": 502
                }
            }, status_code=502)

    async def stream_and_close():
        assistant_full_text = ""
        loop_aborted = False
        try:
            async for line in response.aiter_lines():
                line_str = line if isinstance(line, str) else line.decode("utf-8", errors="ignore")
                
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    try:
                        data_json = json.loads(line_str[6:])
                        choices = data_json.get("choices", [])
                        delta = ""
                        if choices and isinstance(choices, list):
                            fc = choices[0]
                            if isinstance(fc, dict):
                                if "delta" in fc and isinstance(fc["delta"], dict):
                                    delta = fc["delta"].get("content", "")
                                elif "text" in fc:
                                    delta = fc.get("text", "")
                        
                        if delta:
                            assistant_full_text += delta
                            if len(assistant_full_text) > 100 and calculate_ngram_repetition(assistant_full_text, n=3) > 0.6:
                                loop_aborted = True
                                break
                    except Exception:
                        pass
                
                if loop_aborted:
                    break
                
                yield (line_str + "\n").encode("utf-8")

            if loop_aborted:
                await metrics.increment("loops")
                logger.warning(f"[Loop Detector] SSE generation aborted for session {session_id}")
                yield b'data: {"error": {"message": "Generation aborted by anti-entropy loop protection.", "code": 500}}\n\n'
                yield b'data: [DONE]\n\n'
        except Exception as e:
            logger.error(f"[Stream Error] {e}")
        finally:
            await response.aclose()
            if assistant_full_text and not loop_aborted:
                is_dup = await session_entry.check_loop_async(assistant_full_text)
                if is_dup or calculate_ngram_repetition(assistant_full_text, n=3) > 0.55:
                    await metrics.increment("loops")

    resp_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}

    return StreamingResponse(
        stream_and_close(),
        status_code=response.status_code,
        headers=resp_headers
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)