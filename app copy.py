import os
import io
import time
import re
from uuid import uuid4
from typing import Optional, Tuple, List
import base64
import torch
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, g
from flask_cors import CORS
from TTS.utils.radam import RAdam  # 👈 required for newer PyTorch versions

 


# ==== PyTorch safety patch (for 2.6+) ====
torch.serialization.add_safe_globals([RAdam, dict])

# ==== Flask setup ====
app = Flask(__name__)
"""Flask application for Plattdeutsch TTS."""

# ==== Environment configuration ====
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
USE_S3 = os.getenv("USE_S3", "false").lower() in {"1", "true", "yes"}
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "plattdeutsch-tts-audio")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "20000"))
MAX_TEXT_WORDS = int(os.getenv("MAX_TEXT_WORDS", "2000"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
FORCE_JSON_RESPONSE = os.getenv("FORCE_JSON_RESPONSE", "false").lower() in {"1","true","yes"}

# Enable CORS for cross-origin calls from Open WebUI or other UIs
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS.split(",") if CORS_ORIGINS != "*" else "*"}})

# ==== Globals for lazy initialization ====
_tts_instance = None
_tts_samplerate: Optional[int] = None
_s3_client = None
_ip_hits = {}

from collections import deque


# ==== AWS S3 Helper ====
def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return fwd or (request.remote_addr or "-")


@app.before_request
def _before():
    g._t0 = time.perf_counter()
    # Simple per-IP sliding window rate limiter for /synthesize only
    if request.path == "/synthesize" and request.method == "POST":
        now = time.time()
        ip = _client_ip()
        dq = _ip_hits.get(ip)
        if dq is None:
            dq = deque()
            _ip_hits[ip] = dq
        # drop old entries
        while dq and dq[0] <= now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            return jsonify({"error": "Zu viele Anfragen. Bitte kurz warten."}), 429
        dq.append(now)


@app.after_request
def _after(resp):
    try:
        dt_ms = int((time.perf_counter() - getattr(g, "_t0", time.perf_counter())) * 1000)
        print(f"{_client_ip()} {request.method} {request.path} -> {resp.status_code} ({dt_ms} ms)")
    except Exception:
        pass
    return resp


# ==== Load Local Coqui Model (no download needed) ====
def _load_tts() -> Tuple[object, int]:
    """Load and cache the TTS model from local paths."""
    global _tts_instance, _tts_samplerate
    if _tts_instance is not None:
        return _tts_instance, _tts_samplerate

    from TTS.api import TTS  # imported here to save load time

    print("🚀 Loading Coqui TTS model from local paths…")

    # Local paths (inside /models)
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models", "tts_models--de--thorsten--tacotron2-DDC")
    vocoder_dir = os.path.join(base_dir, "models", "vocoder_models--de--thorsten--hifigan_v1")

    model_path = os.path.join(model_dir, "model_file.pth")
    config_path = os.path.join(model_dir, "config.json")
    vocoder_path = os.path.join(vocoder_dir, "model_file.pth.tar")
    vocoder_config_path = os.path.join(vocoder_dir, "config.json")

    # If any required file is missing, attempt auto-download into ./models
    required = [model_path, config_path, vocoder_path, vocoder_config_path]
    if not all(os.path.exists(p) for p in required):
        print("ℹ️  Local models missing. Downloading Thorsten model + vocoder into ./models …")
        try:
            from TTS.utils.manage import ModelManager
            manager = ModelManager()
            manager.output_prefix = os.path.join(base_dir, "models")
            manager.download_model("tts_models/de/thorsten/tacotron2-DDC")
            manager.download_model("vocoder_models/de/thorsten/hifigan_v1")
        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            raise

    # Initialize local model
    tts = TTS(
        model_path=model_path,
        config_path=config_path,
        vocoder_path=vocoder_path,
        vocoder_config_path=vocoder_config_path
    )

    # Ensure attribute expected by newer API exists for older single-speaker models
    try:
        getattr(tts, "is_multi_lingual")
    except AttributeError:
        try:
            object.__setattr__(tts, "is_multi_lingual", False)
        except Exception:
            pass

    # Model initialized


    # Determine sample rate
    sr = 22050
    try:
        sr = getattr(tts, "output_sample_rate", 22050)
    except Exception:
        pass

    _tts_instance = tts
    _tts_samplerate = sr
    print("✅ Model loaded successfully.")
    return _tts_instance, _tts_samplerate


# ==== Convert WAV → MP3 (in-memory) ====
def _wav_to_mp3_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    import lameenc

    # Ensure mono int16 PCM little-endian
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16).tobytes()

    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(1)
    enc.set_quality(2)

    mp3_data = enc.encode(pcm16) + enc.flush()
    return mp3_data


def _normalize_text(text: str) -> str:
    # Normalize whitespace
    text = text.replace("\r", " ").replace("\n", " ")
    # Replace common typographic punctuation with ASCII equivalents
    replacements = {
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201A": ",",  # ‚
        "\u201C": '"',  # “
        "\u201D": '"',  # ”
        "\u00AB": '"',  # «
        "\u00BB": '"',  # »
        "\u2013": "-",  # – en dash
        "\u2014": "-",  # — em dash
        "\u2026": "...",  # … ellipsis
        "\u00A0": " ",  # non-breaking space
    }
    text = text.translate(str.maketrans(replacements))
    # Collapse leftover multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_text_into_chunks(text: str, max_chars: int = 220, min_chars: int = 20) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!?…;:])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_chars and len(merged[-1]) + 1 + len(c) <= max_chars:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    return merged


def _tts_chunk(tts, text: str):
    try:
        return tts.tts(text)
    except AttributeError as e:
        if "is_multi_lingual" in str(e) and hasattr(tts, "synthesizer"):
            return tts.synthesizer.tts(text)
        raise


# ==== Synthesize text → MP3 ====
def _synthesize_to_mp3(text: str) -> Tuple[str, bytes]:
    tts, sr = _load_tts()
    text = _normalize_text(text)
    chunks = _split_text_into_chunks(text)
    if not chunks:
        raise ValueError("Leerer Text nach Normalisierung.")

    silence = np.zeros(int(sr * 0.15), dtype=np.float32)
    wavs: List[np.ndarray] = []
    for c in chunks:
        if len(c) < 3:
            continue
        audio = _tts_chunk(tts, c)
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        wavs.append(audio.astype(np.float32))
        wavs.append(silence)

    if not wavs:
        raise ValueError("Kein Audio erzeugt (zu kurze Segmente?).")

    full = np.concatenate(wavs)
    mp3_bytes = _wav_to_mp3_bytes(full, sr)
    file_id = uuid4().hex[:8]
    filename = f"tts_{int(time.time())}_{file_id}.mp3"
    return filename, mp3_bytes


# ==== Routes ====
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/synthesize")
def synthesize():
    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Bitte Text eingeben."}), 400
    # Validate by words and characters
    norm_for_len = _normalize_text(text)
    word_count = len(norm_for_len.split())
    if word_count > MAX_TEXT_WORDS:
        return jsonify({"error": f"Text ist zu lang (max. {MAX_TEXT_WORDS} Wörter)."}), 400
    if len(norm_for_len) > MAX_TEXT_LEN:
        return jsonify({"error": f"Text ist zu lang (max. {MAX_TEXT_LEN} Zeichen)."}), 400

    try:
        filename, mp3_bytes = _synthesize_to_mp3(text)
    except Exception as e:
        import traceback
        print("❌ Exception during TTS synthesis:")
        traceback.print_exc()  # 👈 This prints full details to PowerShell
        return jsonify({"error": f"TTS-Fehler: {e}"}), 500

    # === Upload to S3 if enabled ===
    if USE_S3 and S3_BUCKET_NAME:
        try:
            s3 = _get_s3_client()
            key = f"audio/{filename}"
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=key,
                Body=mp3_bytes,
                ContentType="audio/mpeg",
            )
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET_NAME, "Key": key},
                ExpiresIn=3600,
            )
            return jsonify({"url": url, "key": key, "filename": filename})
        except Exception as e:
            return jsonify({"error": f"S3-Fehler: {e}"}), 500

    # === Fallback (no S3): return JSON or direct file ===
    wants_json = FORCE_JSON_RESPONSE or ("application/json" in (request.headers.get("Accept") or "")) or (request.args.get("json") in {"1","true","yes"})
    if wants_json:
        b64 = base64.b64encode(mp3_bytes).decode("ascii")
        return jsonify({"filename": filename, "mime": "audio/mpeg", "data_b64": b64})

    return send_file(io.BytesIO(mp3_bytes), mimetype="audio/mpeg", as_attachment=True, download_name=filename)


@app.get("/health")
def health():
    loaded = _tts_instance is not None
    return jsonify(
        {
            "status": "ok",
            "model_loaded": loaded,
            "sample_rate": _tts_samplerate,
            "use_s3": USE_S3,
            "bucket": S3_BUCKET_NAME if USE_S3 else None,
            "region": AWS_REGION,
        }
    )


# ==== Run locally ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

# Optional: avoid favicon 404 noise during local testing
@app.get("/favicon.ico")
def favicon():
    return ("", 204)
