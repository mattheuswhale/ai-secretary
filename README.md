# 7c - AI Secretary using llms (EN / JA / KO)

A fully local, multilingual voice assistant.  
Speak → Whisper STT → Ollama LLM → Kokoro TTS → Speaks back.  
Supports **English**, **Japanese**, and **Korean** with automatic language detection.

---

## Architecture

```
Microphone
    │
    ▼
faster-whisper          (STT: auto-detects EN / JA / KO)
    │
    ▼
Ollama  ◄──────────────────────────────────────────────┐
(gemma4:e2b)                                           │
    │                                                  │
    ├──► get_weather tool ──► Open-Meteo API           │
    │    (on explicit weather questions only)          │
    │                                                  │
    └──► web_search tool ──► DuckDuckGo (ddgs)         │
         (always available; model decides when to use) │
    │                                                  │
    ▼                                                  │
Tool results ──────────────────────────────────────────┘
    │
    ▼
Kokoro TTS              (per-language voice, streamed)
    │
    ▼
sounddevice playback
```

---

## Prerequisites

### System packages

**Linux**
```bash
sudo apt install portaudio19-dev ffmpeg espeak-ng
```

**macOS**
```bash
brew install portaudio ffmpeg espeak-ng
```

**Windows**
- [PortAudio](http://www.portaudio.com/download.html)
- [FFmpeg](https://ffmpeg.org/download.html)
- [eSpeak-NG](https://github.com/espeak-ng/espeak-ng/releases) : download and run the `.msi` installer

### Ollama

Install Ollama from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull gemma4:e2b # or any other llm of your choice
ollama serve          # start the server (may already run as a service)
```

Any model that supports function/tool calling works. Better multilingual models:

```bash
ollama pull qwen3.5:4b    # stronger JA/KO
ollama pull gemma4:e2b     # good balance
```

---

## Installation

### 1. Clone and create a virtual environment

```bash
git clone git@github.com:mattheuswhale/ai-secretary.git
cd voice-assistant

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Japanese support

Japanese G2P (grapheme-to-phoneme) requires `fugashi` and the UniDic dictionary.  
`fugashi[unidic]` installs the package but the dictionary data must be downloaded separately:

```bash
pip install 'fugashi[unidic]'
python -m unidic download
```

> **Note:** The full UniDic 3.1.0 is ~770 MB on disk. If you prefer a smaller footprint and don't need the latest dictionary data, use `unidic-lite` (~250 MB) instead, no download step required:
> ```bash
> pip install 'fugashi[unidic-lite]'
> ```

### 4. Configure your environment

Copy the example file and edit it:

```bash
cp .env.example .env
```

Open `.env` and adjust any values you want to change. At minimum, set:

- `OLLAMA_MODEL` = the model you pulled
- `DEFAULT_CITY` = your city for default weather lookups
- `DEFAULT_LAT` / `DEFAULT_LON` = your coordinates

The assistant runs fine with all defaults if you just want to try it quickly.

---

## Usage

```bash
# Normal run
python assistant.py

# Disable TTS output (text-only, useful for testing)
python assistant.py --no-tts

# Override Ollama model from CLI
python assistant.py --model qwen3.5:4b

# Use a larger Whisper model for noisier environments
python assistant.py --whisper small
python assistant.py --whisper medium
```

---

## Configuration reference

All settings live in `.env`. The file is never to be committed (see `.gitignore`).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma4:e2b` | Model name as shown in `ollama list` |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE` | `int8` | `int8` / `float16` / `float32` |
| `SILENCE_THRESHOLD` | `0.015` | RMS floor before calibration (rarely needs changing) |
| `SILENCE_SECONDS` | `1.5` | Seconds of silence to end a recording |
| `MAX_RECORD_SECS` | `30` | Hard recording cutoff |
| `VAD_CALIBRATION_SECS` | `1.0` | How long to sample ambient noise at startup |
| `VAD_SPEECH_MULTIPLIER` | `3.0` | Trigger threshold = ambient × this. Raise if ambient noise triggers recording. |
| `VAD_MIN_SPEECH_SECS` | `0.4` | Minimum continuous speech before committing. Raise to ignore brief sounds. |
| `KOKORO_VOICE_EN` | `af_heart` | English voice name |
| `KOKORO_VOICE_JA` | `jf_alpha` | Japanese voice name |
| `KOKORO_VOICE_KO` | `kf_nayeon` | Korean voice name |
| `DEFAULT_LAT` | `51.4778` | Fallback latitude for weather (Greenwich) |
| `DEFAULT_LON` | `0.0015` | Fallback longitude for weather (Greenwich) |
| `DEFAULT_CITY` | `Greenwich` | Fallback city name for weather |

To list available Kokoro voices for a language:
```python
from kokoro import KPipeline
p = KPipeline(lang_code='a')   # 'a'=EN, 'j'=JA, 'k'=KO
print(list(p.voices))
```

---

## Tools

The assistant has two built-in agents:

**`get_weather`**: fetches current conditions and a 3-day forecast from [Open-Meteo](https://open-meteo.com/) (free, no API key). Only called when you explicitly ask about weather.

**`web_search`**: queries DuckDuckGo via the `ddgs` library (no API key). Always available; the model calls it when a query requires current or uncertain information.

---

## Troubleshooting

**No audio input / microphone not detected**
```python
import sounddevice
print(sounddevice.query_devices())
```
Set `sd.default.device` in the code or configure your OS default input device.

**VAD triggers on ambient noise**  
Increase `VAD_SPEECH_MULTIPLIER` (e.g. `4.0` or `5.0`) in `.env`.

**First word of sentences is clipped**  
Decrease `VAD_MIN_SPEECH_SECS` (e.g. `0.2`).

**Whisper not detecting Japanese or Korean well**  
Switch to `WHISPER_MODEL=small` or `medium`.

**Tool calls not working (model ignores web_search)**  
Small models like `gemma4:e2b` or `llama3.2:3b` don't invoke tools too reliably. Try `qwen3.5:4b` or `gemma4:e4b` as they handle tool calling more consistently.

**`fugashi` import error on Japanese speech**  
Make sure you ran `python -m unidic download` after installing `fugashi[unidic]`.  
If you skipped this, run it now. It downloads ~770 MB from AWS S3.

**Kokoro TTS produces no audio**  
Verify `espeak-ng` is installed (`espeak-ng --version`). Kokoro uses it as a fallback for unknown phoneme sequences.

---

## Security notes

- `.env` is excluded from git via `.gitignore`. Never commit it.
- Ollama runs on `localhost` only by default. Do not expose port 11434 publicly.
- Web search results are passed to the LLM without additional sanitization. Don't run this in an untrusted network environment.

---

## License

GPLv3   
