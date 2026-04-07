"""
Voice Assistant — Multilingual (EN / JA / KO)
Stack: faster-whisper → Ollama → Kokoro TTS → sounddevice
Weather agent via Open-Meteo (no API key required)
"""

import argparse
import html
import os
import queue
import tempfile
import threading
import sys
import re
import json
import time
from datetime import datetime
import struct
import wave
import io
from pathlib import Path

import numpy as np
import sounddevice as sd

# ── Config (.env) ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; falls back to hardcoded defaults

# ── STT ─────────────────────────────────────────────────────────────────
from faster_whisper import WhisperModel

# ── LLM ─────────────────────────────────────────────────────────────────
import requests as req

# ── TTS ─────────────────────────────────────────────────────────────────
from kokoro import KPipeline
# Note: soundfile is no longer required — Kokoro returns numpy arrays directly.

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  (values loaded from .env, with hardcoded fallbacks)
# ═══════════════════════════════════════════════════════════════════════
SAMPLE_RATE       = 16_000   # Hz — fixed; Whisper requires 16 kHz
CHANNELS          = 1
BLOCK_SIZE        = 1024

SILENCE_THRESHOLD    = float(os.getenv("SILENCE_THRESHOLD",    "0.015"))
SILENCE_SECONDS      = float(os.getenv("SILENCE_SECONDS",      "1.5"))
MAX_RECORD_SECS      = int(  os.getenv("MAX_RECORD_SECS",       "30"))
VAD_CALIBRATION_SECS = float(os.getenv("VAD_CALIBRATION_SECS", "1.0"))
VAD_SPEECH_MULTIPLIER= float(os.getenv("VAD_SPEECH_MULTIPLIER","3.0"))
VAD_MIN_SPEECH_SECS  = float(os.getenv("VAD_MIN_SPEECH_SECS",  "0.4"))

WHISPER_MODEL   = os.getenv("WHISPER_MODEL",   "base")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE",  "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

# Kokoro KPipeline lang_code map
# 'a' = American English, 'j' = Japanese (requires: pip install misaki[ja] + espeak-ng)
# 'k' = Korean (built into misaki, no extra dep)
KOKORO_LANG_MAP = {
    "en": ("a", os.getenv("KOKORO_VOICE_EN", "af_heart")),
    "ja": ("j", os.getenv("KOKORO_VOICE_JA", "jf_alpha")),
    "ko": ("k", os.getenv("KOKORO_VOICE_KO", "kf_nayeon")),
}
KOKORO_LANG_DEFAULT = ("a", os.getenv("KOKORO_VOICE_EN", "af_heart"))

WEATHER_API = "https://api.open-meteo.com/v1/forecast"
GEO_API     = "https://geocoding-api.open-meteo.com/v1/search"

DEFAULT_LAT  = float(os.getenv("DEFAULT_LAT",  "51.4778"))
DEFAULT_LON  = float(os.getenv("DEFAULT_LON",  "0.0015"))
DEFAULT_CITY =       os.getenv("DEFAULT_CITY", "Greenwich")

# ═══════════════════════════════════════════════════════════════════════
# WEATHER AGENT
# ═══════════════════════════════════════════════════════════════════════

def sanitize_city(city: str) -> str:
    """Fix #3: strip non-city characters and cap length before sending to geocoding API."""
    city = re.sub(r"[^\w\s\-\.]", "", city, flags=re.UNICODE).strip()
    return city[:100]


def geocode(city: str) -> tuple[float, float, str]:
    """Return (lat, lon, resolved_name) for a city name."""
    city = sanitize_city(city)
    if not city:
        return DEFAULT_LAT, DEFAULT_LON, DEFAULT_CITY
    try:
        r = req.get(GEO_API, params={"name": city, "count": 1, "language": "en"},
                    timeout=5, verify=True)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            res = results[0]
            return res["latitude"], res["longitude"], res.get("name", city)
    except Exception as e:
        print(f"[geo] {e}")
    return DEFAULT_LAT, DEFAULT_LON, DEFAULT_CITY


def get_weather(city: str | None = None) -> dict:
    """Fetch current weather + short forecast for a city."""
    if city:
        lat, lon, resolved = geocode(city)
    else:
        lat, lon, resolved = DEFAULT_LAT, DEFAULT_LON, DEFAULT_CITY

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "weather_code",
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weather_code",
        ],
        "forecast_days": 3,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    try:
        r = req.get(WEATHER_API, params=params, timeout=8, verify=True)
        r.raise_for_status()
        # Guard against unexpectedly large responses before parsing
        if int(r.headers.get("Content-Length", 0)) > 1_000_000:
            return {"error": "Weather API response too large"}
        data = r.json()
        cur = data["current"]
        daily = data["daily"]

        # WMO weather code → human text (subset)
        wmo_desc = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "icy fog",
            51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
            61: "light rain", 63: "rain", 65: "heavy rain",
            71: "light snow", 73: "snow", 75: "heavy snow",
            80: "rain showers", 81: "rain showers", 82: "violent showers",
            95: "thunderstorm", 96: "thunderstorm with hail",
        }
        def wmo(code):
            return wmo_desc.get(int(code), f"code {code}")

        result = {
            "city": resolved,
            "current": {
                "temp_c": cur["temperature_2m"],
                "feels_like_c": cur["apparent_temperature"],
                "humidity_pct": cur["relative_humidity_2m"],
                "wind_ms": cur["wind_speed_10m"],
                "condition": wmo(cur["weather_code"]),
            },
            "forecast": [
                {
                    "date": daily["time"][i],
                    "max_c": daily["temperature_2m_max"][i],
                    "min_c": daily["temperature_2m_min"][i],
                    "precip_mm": daily["precipitation_sum"][i],
                    "condition": wmo(daily["weather_code"][i]),
                }
                for i in range(3)
            ],
        }
        return result
    except Exception as e:
        return {"error": str(e)}


def weather_tool_call(tool_input: dict) -> str:
    city = tool_input.get("city")
    data = get_weather(city)
    return json.dumps(data, ensure_ascii=False)


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather and 3-day forecast for a city. "
                "ONLY call this tool when the user EXPLICITLY asks about weather, "
                "temperature, rain, snow, wind, forecast, or whether to bring an umbrella. "
                "Do NOT call this for general conversation or any non-weather topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name in English (e.g. 'Seoul', 'Tokyo', 'London'). "
                                       "Omit to use the user's default location.",
                    }
                },
                "required": [],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Keyword gate for weather — tool only included when user mentions weather-related words.
_WEATHER_KEYWORDS = re.compile(
    r"\b(weather|forecast|temperature|temp|rain|snow|wind|storm|cloud|sunny|humid|"
    r"umbrella|jacket|cold|hot|warm|freezing|foggy|typhoon|hurricane|"
    r"날씨|기온|비|눈|바람|흐림|맑음|우산|기상|"
    r"天気|気温|雨|雪|風|晴れ|曇り|傘|予報)\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════
# WEB SEARCH AGENT
# ═══════════════════════════════════════════════════════════════════════

def normalize_query(query: str) -> str:
    # crude heuristic: if non-ascii → let LLM translate via itself
    if not query.isascii():
        return f"Translate to English and search: {query}"
    return query

def web_search(query: str, max_results: int = 4) -> dict:
    """
    Search the web via the ddgs(duckduckgo-search) library (pip install ddgs).
    No API key required. Returns real web results including current events and news.
    """
    from ddgs import DDGS
    query = normalize_query(query)
    query = sanitize_search_query(query)
    if not query:
        return {"error": "empty query"}
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", "")[:200],
                    "snippet": r.get("body",  "")[:400],
                    "url":     r.get("href",  ""),
                })
    except Exception as e:
        print(f"[search] duckduckgo-search error: {e}")
        return {"error": str(e)}
    if not results:
        return {"error": f"No results found for: {query}"}
    return {"query": query, "results": results}


def web_search_tool_call(tool_input: dict) -> str:
    raw = tool_input.get("query", "")

    # Handle broken tool args from small models that pass schema dict instead of string
    if isinstance(raw, dict):
        raw = raw.get("description") or raw.get("value") or ""

    if not isinstance(raw, str):
        raw = str(raw)

    query = normalize_query(raw)
    query = sanitize_search_query(query)

    if not query:
        return json.dumps({"error": "empty query"})

    data = web_search(query)
    return json.dumps(data, ensure_ascii=False)


def sanitize_search_query(query: str) -> str:
    """Strip control characters (not hyphens — valid in queries) and cap length."""
    query = re.sub(r"[\x00-\x1f\x7f]", "", query).strip()
    return query[:200]


# ═══════════════════════════════════════════════════════════════════════
# STT  (faster-whisper)
# ═══════════════════════════════════════════════════════════════════════

class SpeechListener:
    def __init__(self, whisper_model: str = WHISPER_MODEL):
        print(f"[stt] Loading Whisper model '{whisper_model}' …")
        self.model = WhisperModel(whisper_model, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        self.speech_threshold = SILENCE_THRESHOLD  # overwritten by calibrate()
        print("[stt] Ready.")

    def calibrate(self) -> None:
        """Sample ambient noise for VAD_CALIBRATION_SECS and set a dynamic threshold."""
        print("[stt] Calibrating ambient noise … please stay quiet.")
        rms_samples: list[float] = []

        def _cb(indata, frames, time_info, status):
            rms_samples.append(float(np.sqrt(np.mean(indata[:, 0] ** 2))))

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", blocksize=BLOCK_SIZE, callback=_cb):
            time.sleep(VAD_CALIBRATION_SECS + 0.1)

        if rms_samples:
            ambient = float(np.mean(rms_samples))
            self.speech_threshold = max(ambient * VAD_SPEECH_MULTIPLIER, SILENCE_THRESHOLD)
            print(f"[stt] Ambient RMS={ambient:.4f}  →  speech threshold={self.speech_threshold:.4f}")
        else:
            self.speech_threshold = SILENCE_THRESHOLD
            print(f"[stt] Calibration failed, using default threshold={self.speech_threshold:.4f}")

    def record_until_silence(self) -> np.ndarray:
        """Block until speech detected, record until silence, return float32 array."""
        # Bounded queue prevents unbounded RAM use if processing falls behind
        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        recording    = []
        pre_buffer   = []   # chunks collected during the confirmation window
        silence_counter = [0.0]
        speech_counter  = [0.0]   # tracks continuous above-threshold duration
        speech_started  = [False]
        threshold = self.speech_threshold

        def callback(indata, frames, time_info, status):
            chunk = indata[:, 0].copy()
            audio_q.put(chunk)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", blocksize=BLOCK_SIZE,
                            callback=callback):
            print("[stt] Listening … (speak now)")
            start = time.time()
            while True:
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                block_secs = BLOCK_SIZE / SAMPLE_RATE

                if not speech_started[0]:
                    if rms > threshold:
                        # Store chunk in pre_buffer so we don't lose the leading audio
                        # while we wait for VAD_MIN_SPEECH_SECS confirmation.
                        pre_buffer.append(chunk)
                        speech_counter[0] += block_secs
                        if speech_counter[0] >= VAD_MIN_SPEECH_SECS:
                            # Confirmed real speech — flush pre_buffer into recording
                            speech_started[0] = True
                            recording.extend(pre_buffer)
                            pre_buffer.clear()
                            print("[stt] Recording …")
                    else:
                        # Noise spike dropped; clear both counters and the buffer
                        speech_counter[0] = 0.0
                        pre_buffer.clear()
                    continue

                recording.append(chunk)

                if rms < threshold:
                    silence_counter[0] += block_secs
                else:
                    silence_counter[0] = 0.0

                if silence_counter[0] >= SILENCE_SECONDS:
                    break
                if time.time() - start > MAX_RECORD_SECS:
                    print("[stt] Max recording length reached.")
                    break

        return np.concatenate(recording) if recording else np.zeros(SAMPLE_RATE, dtype="float32")

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Returns (text, detected_language)."""
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language=None,        # auto-detect EN / JA / KO
            task="transcribe",
        )
        text = " ".join(s.text for s in segments).strip()
        lang = info.language  # "en", "ja", "ko", …
        return text, lang


# ── Prompt injection guard ───────────────────────────────────────────────
_INJECTION_PATTERNS = re.compile(
    r"(ignore (previous|all|prior) instructions?|"
    r"system prompt|"
    r"you are now|"
    r"act as (an? )?(different|new|other)|"
    r"disregard (your |all )?(previous |prior )?instructions?|"
    r"jailbreak|"
    r"do anything now|"
    r"forget (everything|all)|"
    r"new persona)",
    re.IGNORECASE,
)

def _check_prompt_injection(text: str) -> None:
    """Log a warning if transcribed text looks like a prompt injection attempt."""
    if _INJECTION_PATTERNS.search(text):
        print(f"[security] ⚠️  Possible prompt injection detected in input: {text!r}")


# ═══════════════════════════════════════════════════════════════════════
# LLM  (Ollama)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT="""You are a multilingual voice assistant.

Respond in the SAME language as the user (English, Japanese, or Korean).
Keep replies short and natural (1-3 spoken sentences).
Do not use markdown, lists, or special formatting.

You have access to two tools:
- get_weather: for weather information
- web_search: for up to date or uncertain information

TOOL RULES:
- Use web_search for:
  - current events, news, or recent information
  - anything involving "today", "now", or time-sensitive facts
  - anything you are not fully certain about

- Use get_weather only for weather-related questions

- When calling web_search:
  - ALWAYS pass a plain string query
  - NEVER pass objects or structured data
  - ALWAYS write the query in English
  - Example: {"name": "web_search", "arguments": {"query": "current US president"}}

IMPORTANT:
Tool calls MUST be valid JSON. Do NOT write tool calls as plain text.

ANTI-HALLUCINATION:
Never guess or invent information.
If unsure or the question may need recent data, use web_search.

AFTER web_search:
Summarize the results in 1-3 short sentences.

PRIORITY:
weather - get_weather
uncertain or recent - web_search
otherwise - answer normally"""


class LLMAgent:
    # Cap history to last N turns to prevent memory leak and context bloat
    MAX_HISTORY_TURNS = 20

    def __init__(self, ollama_model: str = OLLAMA_MODEL):
        self.ollama_model = ollama_model
        self.history: list[dict] = []

    def chat(self, user_text: str) -> str:
        # Warn on suspicious prompt-injection patterns in transcribed text
        _check_prompt_injection(user_text)

        self.history.append({"role": "user", "content": user_text})

        # Trim history — keep last N turns (pairs of user+assistant)
        if len(self.history) > self.MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]

        # Tool selection:
        # - get_weather: only when user explicitly mentions weather-related words
        # - web_search: always available; the system prompt + model decides when to call it.
        #   Keeping it in the payload at all times aligns with the aggressive search policy
        #   in the system prompt and avoids a keyword gate silently blocking searches.
        tools = [TOOLS_SPEC[1]]  # web_search always available
        if _WEATHER_KEYWORDS.search(user_text):
            tools.insert(0, TOOLS_SPEC[0])  # get_weather prepended on weather queries

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "model": self.ollama_model,
            "messages": [{
                "role": "system",
                "content": SYSTEM_PROMPT + f"\n\nCurrent date and time: {now}"
            }] + self.history,
            "stream": False,
            "tools": tools,
        }

        # Agentic loop — handle tool calls
        # Wall-clock deadline so a looping model can't stall the assistant forever
        loop_deadline = time.time() + 30.0
        for _ in range(5):
            if time.time() > loop_deadline:
                print("[llm] Agent loop deadline exceeded.")
                return "Sorry, that took too long. Please try again."
            try:
                r = req.post(OLLAMA_URL, json=payload, timeout=60, verify=True)
                r.raise_for_status()
            except Exception as e:
                print(f"[llm error] {e}")
                return "Sorry, I couldn't reach the language model. Please try again."

            resp = r.json()
            msg = resp.get("message", {})
            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "") or ""

            # Case 1: proper tool call (highest priority)
            if tool_calls:
                payload["messages"].append(msg)

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = tc["function"].get("arguments", {})

                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {}

                    print(f"[agent] Calling tool: {fn_name}({fn_args})")

                    if fn_name == "get_weather":
                        result = weather_tool_call(fn_args)
                    elif fn_name == "web_search":
                        result = web_search_tool_call(fn_args)
                    else:
                        result = json.dumps({"error": f"unknown tool {fn_name}"})

                    payload["messages"].append({
                        "role": "tool",
                        "content": result,
                    })

                continue


            # Case 2: fallback (ONLY if tool_calls is empty)
            fake_call_match = re.search(r'web_search\s*\{\s*query\s*:\s*(.+?)\s*\}', content, re.IGNORECASE)

            if fake_call_match:
                raw_query = fake_call_match.group(1)

                raw_query = re.sub(r'[<>\|"]', '', raw_query).strip()

                print(f"[agent] Recovered fake tool call: web_search({raw_query})")

                result = web_search_tool_call({"query": raw_query})

                payload["messages"].append(msg)
                payload["messages"].append({
                    "role": "tool",
                    "content": result,
                })

                continue


            # Case 3: normal response
            reply = content.strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply

        return "Sorry, I couldn't complete that request."


# ═══════════════════════════════════════════════════════════════════════
# TTS  (Kokoro — hexgrad/kokoro, KPipeline)
# ═══════════════════════════════════════════════════════════════════════

class Speaker:
    def __init__(self, no_tts: bool = False):
        self.no_tts = no_tts
        # Lazily loaded pipelines keyed by lang_code: { "a": KPipeline, ... }
        self._pipelines: dict[str, KPipeline] = {}
        if not no_tts:
            print("[tts] Kokoro ready (pipelines load on first use).")

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Return a cached KPipeline for the given Kokoro lang_code."""
        if lang_code not in self._pipelines:
            print(f"[tts] Loading Kokoro pipeline for lang_code='{lang_code}' …")
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
            print(f"[tts] Pipeline '{lang_code}' ready.")
        return self._pipelines[lang_code]

    def speak(self, text: str, lang: str = "en"):
        if self.no_tts or not text:
            print(f"[tts] (skipped) {text}")
            return

        # Cap spoken text to prevent hangs on unexpectedly long LLM replies
        TTS_MAX_CHARS = 1000
        if len(text) > TTS_MAX_CHARS:
            print(f"[tts] Truncating {len(text)} → {TTS_MAX_CHARS} chars.")
            text = text[:TTS_MAX_CHARS]

        lang_code, voice = KOKORO_LANG_MAP.get(lang, KOKORO_LANG_DEFAULT)
        try:
            pipeline = self._get_pipeline(lang_code)
            # Stream and play each chunk as it arrives instead of buffering everything.
            # Avoids memory spikes from np.concatenate on large audio arrays,
            # and the NumPy 2.x __array__ copy keyword incompatibility.
            played_any = False
            for _, _, audio in pipeline(text, voice=voice):
                if hasattr(audio, "numpy"):
                    # torch.Tensor — use native conversion, no copy keyword issue
                    chunk = audio.numpy().astype(np.float32)
                else:
                    # Already a numpy array or array-like
                    chunk = np.asarray(audio, dtype=np.float32)
                if chunk.size == 0:
                    continue
                sd.play(chunk, samplerate=24_000)
                sd.wait()
                played_any = True
            if not played_any:
                print("[tts] No audio generated.")
        except Exception as e:
            print(f"[tts] Error: {e}")
            print(f"[tts] Response: {text}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multilingual Voice Assistant")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--whisper", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size")
    args = parser.parse_args()

    # Parse args BEFORE constructing objects so CLI values take effect
    listener = SpeechListener(whisper_model=args.whisper)
    agent    = LLMAgent(ollama_model=args.model)
    speaker  = Speaker(no_tts=args.no_tts)

    listener.calibrate()

    print("\n══════════════════════════════════════")
    print("  Voice Assistant  (EN / JA / KO)")
    print("  Press Ctrl+C to quit")
    print("══════════════════════════════════════\n")

    while True:
        try:
            # 1. Record
            audio = listener.record_until_silence()

            # 2. Transcribe
            text, lang = listener.transcribe(audio)
            if not text:
                print("[stt] Nothing heard, retrying …")
                continue
            print(f"[you({lang})] {text}")

            # 3. LLM (with weather + web search tools)
            print("[llm] Thinking …")
            reply = agent.chat(text)
            print(f"[assistant] {reply}")

            # 4. Speak
            speaker.speak(reply, lang=lang)

        except KeyboardInterrupt:
            print("\n[assistant] Goodbye!")
            break
        except Exception as e:
            print(f"[error] {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
