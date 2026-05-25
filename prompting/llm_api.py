import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import os

API_KEY = os.environ.get("GLM_API_KEY", "")
API_BASE = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-flash"

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

NVIDIA_MODELS = {
    "deepseek": "deepseek-ai/deepseek-v3.1-terminus",
    "llama": "meta/llama-3.3-70b-instruct",
    "qwen": "qwen/qwen3-next-80b-a3b-instruct",
    "gpt-oss": "openai/gpt-oss-120b",
}

DEEPSEEK_MODELS = {
    "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner",
}

def _make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

_session = _make_session()


def chat_completion(
    model: str = DEFAULT_MODEL,
    messages: list = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    max_retries: int = 15,
):
    print(f"[LLM] Request start: model={model}, messages={len(messages or [])}, max_tokens={max_tokens}, temperature={temperature}", flush=True)
    start_time = time.time()
    if model in DEEPSEEK_MODELS or model.startswith("deepseek-"):
        content, usage = _deepseek_call(model, messages, max_tokens, temperature, max_retries)
    elif model in NVIDIA_MODELS or "/" in model:
        model_id = NVIDIA_MODELS.get(model, model)
        content, usage = _nvidia_call(model_id, messages, max_tokens, temperature, max_retries)
    else:
        content, usage = _zhipu_call(model, messages, max_tokens, temperature, max_retries)
    elapsed = time.time() - start_time
    status = "ok" if content is not None else "failed"
    print(f"[LLM] Request end: model={model}, status={status}, elapsed={elapsed:.1f}s", flush=True)
    return content, usage


def _deepseek_call(model, messages, max_tokens, temperature, max_retries):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(max_retries):
        try:
            print(f"[LLM] DeepSeek attempt {attempt + 1}/{max_retries}: model={model}", flush=True)
            resp = _session.post(f"{DEEPSEEK_API_BASE}/chat/completions", headers=headers, json=payload, timeout=(10, 300))
            if resp.status_code == 429:
                wait = min(15 + attempt * 10, 60)
                print(f"[LLM] DeepSeek rate limited (attempt {attempt+1}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"[LLM] DeepSeek API returned {resp.status_code}: {resp.text[:200]}", flush=True)
                time.sleep(5)
                continue
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return content, data.get("usage", {})
        except Exception as e:
            wait = min(5 + attempt * 3, 30)
            print(f"[LLM] DeepSeek error (attempt {attempt+1}): {e}, retrying in {wait}s...", flush=True)
            time.sleep(wait)
    return None, {}


def _zhipu_call(model, messages, max_tokens, temperature, max_retries):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(max_retries):
        try:
            print(f"[LLM] Zhipu attempt {attempt + 1}/{max_retries}: model={model}", flush=True)
            resp = _session.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=(10, 1200))
            if resp.status_code == 429:
                wait = min(30 + attempt * 15, 120)
                print(f"[LLM] Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"[LLM] API returned {resp.status_code}: {resp.text[:200]}", flush=True)
                time.sleep(10)
                continue
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return content, data.get("usage", {})
        except Exception as e:
            wait = min(10 + attempt * 5, 60)
            print(f"[LLM] Request error (attempt {attempt+1}): {e}, retrying in {wait}s...", flush=True)
            time.sleep(wait)
    return None, {}


def _nvidia_call(model, messages, max_tokens, temperature, max_retries):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(3):
        try:
            print(f"[LLM] NVIDIA attempt {attempt + 1}/3: model={model}", flush=True)
            resp = _session.post(f"{NVIDIA_API_BASE}/chat/completions", headers=headers, json=payload, timeout=(10, 120))
            if resp.status_code == 429:
                wait = min(30 + attempt * 15, 60)
                print(f"[LLM] NVIDIA rate limited (attempt {attempt+1}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"[LLM] NVIDIA API returned {resp.status_code}: {resp.text[:200]}", flush=True)
                time.sleep(5)
                continue
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return content, data.get("usage", {})
        except Exception as e:
            print(f"[LLM] NVIDIA request error (attempt {attempt+1}/3): {e}", flush=True)
            time.sleep(3)
    print("[LLM] NVIDIA API failed after 3 attempts, falling back to ZhipuAI glm-4-flash", flush=True)
    return _zhipu_call(DEFAULT_MODEL, messages, max_tokens, temperature, max_retries)
