import os
import io
import time
import re
from uuid import uuid4
from typing import Optional, Tuple, List
import base64
import torch
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, g, redirect, url_for, make_response, abort
import shutil
import json
import zipfile
import tarfile
import tempfile
from flask_cors import CORS
import threading
import psutil
from TTS.utils.radam import RAdam  # 👈 required for newer PyTorch versions

 


# ==== PyTorch safety patch (for 2.6+) ====
torch.serialization.add_safe_globals([RAdam, dict])

# ==== Flask setup ====
app = Flask(__name__)
"""Flask application for Plattdeutsch TTS."""
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024

# ==== Configuration (defaults → persistent config → env overrides) ====
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Defaults
AWS_REGION = "eu-central-1"
USE_S3 = False
S3_BUCKET_NAME = "plattdeutsch-tts-audio"
MAX_TEXT_LEN = 20000
MAX_TEXT_WORDS = 2000
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60  # seconds
CORS_ORIGINS = "*"
FORCE_JSON_RESPONSE = False
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

def _load_persistent_config():
    global USE_S3, S3_BUCKET_NAME, FORCE_JSON_RESPONSE
    global MAX_TEXT_WORDS, MAX_TEXT_LEN, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW
    global CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, SILENCE_SEC
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            USE_S3 = data.get("USE_S3", USE_S3)
            S3_BUCKET_NAME = data.get("S3_BUCKET_NAME", S3_BUCKET_NAME)
            FORCE_JSON_RESPONSE = data.get("FORCE_JSON_RESPONSE", FORCE_JSON_RESPONSE)
            MAX_TEXT_WORDS = data.get("MAX_TEXT_WORDS", MAX_TEXT_WORDS)
            MAX_TEXT_LEN = data.get("MAX_TEXT_LEN", MAX_TEXT_LEN)
            RATE_LIMIT_MAX = data.get("RATE_LIMIT_MAX", RATE_LIMIT_MAX)
            RATE_LIMIT_WINDOW = data.get("RATE_LIMIT_WINDOW", RATE_LIMIT_WINDOW)
            CHUNK_MAX_CHARS = data.get("CHUNK_MAX_CHARS", CHUNK_MAX_CHARS)
            CHUNK_MIN_CHARS = data.get("CHUNK_MIN_CHARS", CHUNK_MIN_CHARS)
            SILENCE_SEC = data.get("SILENCE_SEC", SILENCE_SEC)
        except Exception as e:
            print(f"⚠️ Failed to load config.json: {e}")


def _apply_env_overrides():
    global AWS_REGION, USE_S3, S3_BUCKET_NAME, FORCE_JSON_RESPONSE
    global MAX_TEXT_WORDS, MAX_TEXT_LEN, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW
    global CORS_ORIGINS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, SILENCE_SEC
    AWS_REGION = os.getenv("AWS_REGION", AWS_REGION)
    USE_S3 = os.getenv("USE_S3", str(USE_S3)).lower() in {"1", "true", "yes"}
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", S3_BUCKET_NAME)
    MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", str(MAX_TEXT_LEN)))
    MAX_TEXT_WORDS = int(os.getenv("MAX_TEXT_WORDS", str(MAX_TEXT_WORDS)))
    RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", str(RATE_LIMIT_MAX)))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", str(RATE_LIMIT_WINDOW)))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", CORS_ORIGINS)
    FORCE_JSON_RESPONSE = os.getenv("FORCE_JSON_RESPONSE", str(FORCE_JSON_RESPONSE)).lower() in {"1","true","yes"}
    CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", str(CHUNK_MAX_CHARS if 'CHUNK_MAX_CHARS' in globals() else 220)))
    CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", str(CHUNK_MIN_CHARS if 'CHUNK_MIN_CHARS' in globals() else 20)))
    SILENCE_SEC = float(os.getenv("SILENCE_SEC", str(SILENCE_SEC if 'SILENCE_SEC' in globals() else 0.15)))


# Initialize from persistent config then env overrides
CHUNK_MAX_CHARS = 220
CHUNK_MIN_CHARS = 20
SILENCE_SEC = 0.15
_load_persistent_config()
_apply_env_overrides()

# Enable CORS for cross-origin calls from Open WebUI or other UIs
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS.split(",") if CORS_ORIGINS != "*" else "*"}})


# ==== Template globals (for base layout & nav) ====
@app.context_processor
def _inject_template_globals():
    try:
        path = request.path or "/"
        admin_cookie = request.cookies.get("admin_token") or ""
        admin_logged_in = bool(ADMIN_TOKEN) and admin_cookie == ADMIN_TOKEN
    except Exception:
        path = "/"
        admin_logged_in = False
    return {
        "is_admin_route": path.startswith("/admin"),
        "admin_logged_in": admin_logged_in,
    }

# ==== Globals for lazy initialization ====
_tts_instance = None
_tts_samplerate: Optional[int] = None
_s3_client = None
_ip_hits = {}
_auto_downloaded = False
START_TIME = time.time()
_model_lock = threading.Lock()
_download_status = {"state": "idle", "message": "", "percent": 0}

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
    """Load and cache the TTS model from local paths.
    Respects optional .active_model.json to choose model folders.
    """
    global _tts_instance, _tts_samplerate
    if _tts_instance is not None:
        return _tts_instance, _tts_samplerate

    from TTS.api import TTS  # imported here to save load time

    print("🚀 Loading Coqui TTS model from local paths…")

    # Local paths (inside /models) and optional active selection
    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    active_file = os.path.join(base_dir, ".active_model.json")
    active_tts = None
    active_vocoder = None
    try:
        if os.path.exists(active_file):
            with open(active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                active_tts = data.get("tts") or None
                active_vocoder = data.get("vocoder") or None
    except Exception:
        pass

    model_dir = os.path.join(models_root, active_tts) if active_tts else os.path.join(models_root, "tts_models--de--thorsten--tacotron2-DDC")
    vocoder_dir = os.path.join(models_root, active_vocoder) if active_vocoder else os.path.join(models_root, "vocoder_models--de--thorsten--hifigan_v1")

    model_path = os.path.join(model_dir, "model_file.pth")
    config_path = os.path.join(model_dir, "config.json")
    vocoder_path = os.path.join(vocoder_dir, "model_file.pth.tar")
    vocoder_config_path = os.path.join(vocoder_dir, "config.json")

    # Production-safe behavior: do NOT auto-download. Require explicit admin action.
    required = [model_path, config_path, vocoder_path, vocoder_config_path]
    if not all(os.path.exists(p) for p in required):
        missing = [p for p in required if not os.path.exists(p)]
        friendly = ", ".join([os.path.relpath(p, start=os.path.dirname(__file__)) for p in missing])
        raise RuntimeError(
            f"Kein Modell installiert oder unvollständig. Fehlende Dateien: {friendly}. Bitte Modelle im Admin‑Bereich herunterladen/importieren."
        )

    # Initialize local model
    with _model_lock:
        if _tts_instance is not None:
            return _tts_instance, _tts_samplerate
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


def _split_text_into_chunks(text: str, max_chars: int = None, min_chars: int = None) -> List[str]:
    if max_chars is None:
        max_chars = CHUNK_MAX_CHARS
    if min_chars is None:
        min_chars = CHUNK_MIN_CHARS
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

    silence = np.zeros(int(sr * SILENCE_SEC), dtype=np.float32)
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
    return render_template(
        "index.html",
        limits={
            "words": MAX_TEXT_WORDS,
            "chars": MAX_TEXT_LEN,
        },
    )


@app.get("/impressum")
def impressum():
    return render_template("impressum.html")


@app.get("/info")
def info_page():
    return render_template("info.html")


@app.get("/admin-info")
def admin_info_page():
    return render_template("admin_info.html")


@app.get("/logo.png")
def logo_png():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "logo.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    abort(404)


@app.get("/login")
def login_page():
    return render_template("login.html", title="Admin Anmelden")


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
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
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
    # Collect versions without importing heavy modules if possible
    try:
        from importlib.metadata import version as _v
    except Exception:  # pragma: no cover
        _v = None
    def ver(name: str):
        try:
            return _v(name) if _v else None
        except Exception:
            return None

    # Active model selection
    base_dir = os.path.dirname(__file__)
    active_file = os.path.join(base_dir, ".active_model.json")
    active = {"tts": None, "vocoder": None}
    try:
        if os.path.exists(active_file):
            with open(active_file, "r", encoding="utf-8") as f:
                active = json.load(f) or active
    except Exception:
        pass

    # Determine if active models present on disk
    models_present = False
    try:
        base_dir = os.path.dirname(__file__)
        models_root = os.path.join(base_dir, "models")
        tts_name = active.get("tts")
        voc_name = active.get("vocoder")
        if tts_name and voc_name:
            tts_dir = os.path.join(models_root, tts_name)
            voc_dir = os.path.join(models_root, voc_name)
            required = [
                os.path.join(tts_dir, "model_file.pth"),
                os.path.join(tts_dir, "config.json"),
                os.path.join(voc_dir, "model_file.pth.tar"),
                os.path.join(voc_dir, "config.json"),
            ]
            models_present = all(os.path.exists(p) for p in required)
    except Exception:
        models_present = False

    data = {
        "status": "ok",
        "model_loaded": loaded,
        "model_name": "tts_models/de/thorsten/tacotron2-DDC",
        "sample_rate": _tts_samplerate,
        "use_s3": USE_S3,
        "bucket": S3_BUCKET_NAME if USE_S3 else None,
        "region": AWS_REGION,
        "active_models": active,
        "models_present": bool(models_present),
        "uptime_seconds": int(time.time() - START_TIME),
        "auto_downloaded": bool(_auto_downloaded),
        "limits": {
            "max_text_words": MAX_TEXT_WORDS,
            "max_text_len": MAX_TEXT_LEN,
            "rate_limit_max": RATE_LIMIT_MAX,
            "rate_limit_window": RATE_LIMIT_WINDOW,
        },
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "min_chars": CHUNK_MIN_CHARS,
            "silence_sec": SILENCE_SEC,
        },
        "cors": {
            "origins": CORS_ORIGINS,
            "force_json": FORCE_JSON_RESPONSE,
        },
        "versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "Flask": ver("Flask"),
            "TTS": ver("TTS"),
            "torch": ver("torch"),
            "numpy": ver("numpy"),
            "boto3": ver("boto3"),
        },
    }
    return jsonify(data)


# ==== Admin helpers ====
def _admin_token() -> str:
    return ADMIN_TOKEN or ""


def _is_admin_request() -> bool:
    token = _admin_token()
    if not token:
        return False
    supplied = (
        request.args.get("token")
        or request.headers.get("X-Admin-Token")
        or request.cookies.get("admin_token")
        or ""
    )
    return supplied == token


def _set_admin_cookie_if_needed(resp):
    token = _admin_token()
    if not token:
        return resp
    supplied = request.args.get("token") or request.headers.get("X-Admin-Token")
    if supplied == token and request.cookies.get("admin_token") != token:
        resp.set_cookie(
            "admin_token",
            token,
            max_age=7 * 24 * 3600,
            httponly=True,
            samesite="Lax",
        )
    return resp


@app.get("/admin")
def admin_page():
    if not _admin_token():
        return (
            "Admin ist deaktiviert. Setzen Sie die Umgebungsvariable ADMIN_TOKEN.",
            403,
        )
    if not _is_admin_request():
        return ("Nicht autorisiert. Fügen Sie ?token=… hinzu oder Header X-Admin-Token.", 403)

    # Model folders
    base_dir = os.path.dirname(__file__)
    tts_dir = os.path.join(base_dir, "models", "tts_models--de--thorsten--tacotron2-DDC")
    voc_dir = os.path.join(base_dir, "models", "vocoder_models--de--thorsten--hifigan_v1")

    # Discover available model folders
    available = []
    def _read_model_meta(folder_path: str):
        try:
            with open(os.path.join(folder_path, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {
                "language": ((cfg.get("model") or {}).get("language") or "-"),
                "type": ((cfg.get("model") or {}).get("type") or "-"),
                "sample_rate": ((cfg.get("audio") or {}).get("sample_rate") or "-"),
            }
        except Exception:
            return {"language": "-", "type": "-", "sample_rate": "-"}

    try:
        for name in sorted(os.listdir(os.path.join(base_dir, "models"))):
            if not (name.startswith("tts_models--") or name.startswith("vocoder_models--")):
                continue
            mtype = "TTS" if name.startswith("tts_models--") else "Vocoder"
            meta = _read_model_meta(os.path.join(base_dir, "models", name))
            available.append({"name": name, "type": mtype, "meta": meta})
    except Exception:
        pass

    # Load active selection
    active_file = os.path.join(base_dir, ".active_model.json")
    active = {"tts": None, "vocoder": None}
    try:
        if os.path.exists(active_file):
            with open(active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                active["tts"] = data.get("tts")
                active["vocoder"] = data.get("vocoder")
    except Exception:
        pass

    # Mark active in list
    for m in available:
        m["active"] = (m["name"] == active.get("tts") or m["name"] == active.get("vocoder"))

    # System info
    def _system_info():
        try:
            mem = psutil.virtual_memory()
            # Use filesystem root for disk usage
            root_path = os.path.abspath(os.sep)
            disk = psutil.disk_usage(root_path)
            cpu = psutil.cpu_percent(interval=0.2)
            uptime = int(time.time() - START_TIME)
            return {
                "cpu": cpu,
                "ram": mem.percent,
                "disk_free": round(100 - disk.percent, 1),
                "uptime": uptime,
            }
        except Exception:
            return {"cpu": None, "ram": None, "disk_free": None, "uptime": int(time.time() - START_TIME)}

    # Compute missing files for currently active pair (if any)
    def _active_missing_list():
        try:
            models_root = os.path.join(base_dir, "models")
            tts_name = active.get("tts")
            voc_name = active.get("vocoder")
            if not (tts_name and voc_name):
                return []
            tts_dir_a = os.path.join(models_root, tts_name)
            voc_dir_a = os.path.join(models_root, voc_name)
            req = [
                os.path.join(tts_dir_a, "model_file.pth"),
                os.path.join(tts_dir_a, "config.json"),
                os.path.join(voc_dir_a, "model_file.pth.tar"),
                os.path.join(voc_dir_a, "config.json"),
            ]
            return [os.path.relpath(p, start=base_dir) for p in req if not os.path.exists(p)]
        except Exception:
            return []

    ctx = {
        "use_s3": USE_S3,
        "bucket": S3_BUCKET_NAME,
        "region": AWS_REGION,
        "force_json": FORCE_JSON_RESPONSE,
        "limits": {
            "words": MAX_TEXT_WORDS,
            "chars": MAX_TEXT_LEN,
            "rate_max": RATE_LIMIT_MAX,
            "rate_window": RATE_LIMIT_WINDOW,
        },
        "chunking": {
            "max_chars": CHUNK_MAX_CHARS,
            "min_chars": CHUNK_MIN_CHARS,
            "silence_sec": SILENCE_SEC,
        },
        "model_loaded": _tts_instance is not None,
        "models": {
            "tts_dir": tts_dir,
            "tts_exists": os.path.isdir(tts_dir),
            "vocoder_dir": voc_dir,
            "vocoder_exists": os.path.isdir(voc_dir),
            "available": available,
            "active": active,
        },
        "message": request.args.get("msg") or "",
        "cors": {
            "origins": CORS_ORIGINS,
            "force_json": FORCE_JSON_RESPONSE,
        },
        "download_status": _download_status,
        "system": _system_info(),
        "active_missing": _active_missing_list(),
    }
    resp = make_response(render_template("admin.html", **ctx))
    return _set_admin_cookie_if_needed(resp)


@app.post("/admin/save")
def admin_save():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    global USE_S3, S3_BUCKET_NAME, FORCE_JSON_RESPONSE
    global MAX_TEXT_WORDS, MAX_TEXT_LEN, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW
    global CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, SILENCE_SEC

    USE_S3 = bool(request.form.get("use_s3"))
    S3_BUCKET_NAME = (request.form.get("bucket") or S3_BUCKET_NAME).strip()
    FORCE_JSON_RESPONSE = bool(request.form.get("force_json"))

    def _int(name, default):
        try:
            return int(request.form.get(name, default))
        except Exception:
            return default

    def _float(name, default):
        try:
            return float(request.form.get(name, default))
        except Exception:
            return default

    MAX_TEXT_WORDS = _int("max_words", MAX_TEXT_WORDS)
    MAX_TEXT_LEN = _int("max_chars", MAX_TEXT_LEN)
    RATE_LIMIT_MAX = _int("rate_max", RATE_LIMIT_MAX)
    RATE_LIMIT_WINDOW = _int("rate_window", RATE_LIMIT_WINDOW)
    CHUNK_MAX_CHARS = _int("chunk_max", CHUNK_MAX_CHARS)
    CHUNK_MIN_CHARS = _int("chunk_min", CHUNK_MIN_CHARS)
    SILENCE_SEC = _float("silence", SILENCE_SEC)

    # Persist to config.json
    data = {
        "USE_S3": USE_S3,
        "S3_BUCKET_NAME": S3_BUCKET_NAME,
        "FORCE_JSON_RESPONSE": FORCE_JSON_RESPONSE,
        "MAX_TEXT_WORDS": MAX_TEXT_WORDS,
        "MAX_TEXT_LEN": MAX_TEXT_LEN,
        "RATE_LIMIT_MAX": RATE_LIMIT_MAX,
        "RATE_LIMIT_WINDOW": RATE_LIMIT_WINDOW,
        "CHUNK_MAX_CHARS": CHUNK_MAX_CHARS,
        "CHUNK_MIN_CHARS": CHUNK_MIN_CHARS,
        "SILENCE_SEC": SILENCE_SEC,
    }
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write config.json: {e}")

    return redirect(url_for("admin_page", msg="Einstellungen gespeichert."))


@app.post("/admin/flush-limiter")
def admin_flush_limiter():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    _ip_hits.clear()
    return redirect(url_for("admin_page", msg="Rate-Limiter geleert."))


@app.post("/admin/reload-model")
def admin_reload_model():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    global _tts_instance, _tts_samplerate
    with _model_lock:
        _tts_instance = None
        _tts_samplerate = None
    return redirect(url_for("admin_page", msg="Modell wird beim nächsten Aufruf neu geladen."))


@app.post("/admin/clean-models")
def admin_clean_models():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    removed = []
    if os.path.isdir(models_root):
        for name in os.listdir(models_root):
            if name.startswith("tts_models--") or name.startswith("vocoder_models--"):
                try:
                    shutil.rmtree(os.path.join(models_root, name), ignore_errors=True)
                    removed.append(name)
                except Exception:
                    pass
    # Also clear active selection
    active_file = os.path.join(base_dir, ".active_model.json")
    try:
        if os.path.exists(active_file):
            os.remove(active_file)
    except Exception:
        pass
    # Reset loaded model
    global _tts_instance, _tts_samplerate, _auto_downloaded
    with _model_lock:
        _tts_instance = None
        _tts_samplerate = None
        _auto_downloaded = False
    msg = (
        "Alle Modelle gelöscht. Kein Modell installiert." if removed else "Keine Modelle gefunden."
    )
    return redirect(url_for("admin_page", msg=msg))


@app.post("/admin/download-models")
def admin_download_models():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "models")
    os.makedirs(out_path, exist_ok=True)
    global _download_status
    if _download_status.get("state") == "running":
        return redirect(url_for("admin_page", msg="Download läuft bereits…"))

    def _job(root: str):
        global _download_status
        _download_status = {"state": "running", "message": "Modelldownload gestartet …", "percent": 0}
        try:
            from TTS.utils.manage import ModelManager
            manager = ModelManager()
            manager.output_prefix = root
            _download_status = {"state": "running", "message": "Lade TTS‑Modell …", "percent": 10}
            manager.download_model("tts_models/de/thorsten/tacotron2-DDC")
            _download_status = {"state": "running", "message": "Lade Vocoder …", "percent": 55}
            manager.download_model("vocoder_models/de/thorsten/hifigan_v1")
            # Verify integrity of downloaded models
            tts_dir_v = os.path.join(root, "tts_models--de--thorsten--tacotron2-DDC")
            voc_dir_v = os.path.join(root, "vocoder_models--de--thorsten--hifigan_v1")
            checks = [
                os.path.join(tts_dir_v, "model_file.pth"),
                os.path.join(tts_dir_v, "config.json"),
                os.path.join(voc_dir_v, "model_file.pth.tar"),
                os.path.join(voc_dir_v, "config.json"),
            ]
            _download_status = {"state": "running", "message": "Prüfe Dateien …", "percent": 90}
            missing = [os.path.relpath(p, start=os.path.dirname(__file__)) for p in checks if not os.path.exists(p)]
            if missing:
                _download_status = {"state": "error", "message": f"Unvollständiger Download. Fehlend: {', '.join(missing)}", "percent": 0}
            else:
                _download_status = {"state": "done", "message": "Modelle erfolgreich heruntergeladen.", "percent": 100}
        except Exception as e:
            _download_status = {"state": "error", "message": f"Fehler beim Download: {e}", "percent": 0}

    threading.Thread(target=_job, args=(out_path,), daemon=True).start()
    return redirect(url_for("admin_page", msg="Modelldownload gestartet – bitte einige Minuten warten."))


def _install_dir_to_models(src_dir: str, models_root: str) -> str:
    # Detect target subdir
    entries = [d for d in os.listdir(src_dir)]
    # Prefer a folder that already has the normalized name
    target_name = None
    for name in entries:
        if name.startswith("tts_models--") or name.startswith("vocoder_models--"):
            target_name = name
            break
    if not target_name:
        # Heuristic: presence of typical files
        # TTS typically has config.json + model_file.pth
        # Vocoder often has config.json + model_file.pth.tar or generator params
        is_vocoder = False
        if os.path.exists(os.path.join(src_dir, "model_file.pth.tar")):
            is_vocoder = True
        elif any("vocoder" in e.lower() or "hifigan" in e.lower() for e in entries):
            is_vocoder = True
        target_name = (
            "vocoder_models--custom--uploaded" if is_vocoder else "tts_models--custom--uploaded"
        )
        dst = os.path.join(models_root, target_name)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src_dir, dst)
        return f"Installiert nach {dst}"

    # Move/replace detected folder
    src_path = os.path.join(src_dir, target_name)
    dst_path = os.path.join(models_root, target_name)
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path, ignore_errors=True)
    shutil.move(src_path, dst_path)
    return f"Installiert {target_name}"


def _is_safe_member(base_dir: str, member_path: str) -> str:
    """Resolve an archive member path safely within base_dir.
    Prevents directory traversal and absolute path writes.
    Returns the absolute destination path if safe, otherwise raises ValueError.
    """
    base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base, member_path))
    try:
        common = os.path.commonpath([base, target])
    except Exception:
        raise ValueError("Unsicherer Pfad im Archiv.")
    if common != base:
        raise ValueError("Unsicherer Pfad im Archiv (Traversal / absolut).")
    return target


def _extract_archive_to_tmp(archive_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="tts_upload_")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as z:
            for info in z.infolist():
                name = info.filename
                if name.endswith('/'):
                    dest_dir = _is_safe_member(tmp_dir, name)
                    os.makedirs(dest_dir, exist_ok=True)
                    continue
                dest = _is_safe_member(tmp_dir, name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(info) as src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as t:
            for member in t.getmembers():
                if member.issym() or member.islnk():
                    continue
                name = member.name
                dest = _is_safe_member(tmp_dir, name)
                if member.isdir():
                    os.makedirs(dest, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    src = t.extractfile(member)
                    if src:
                        with open(dest, 'wb') as out:
                            shutil.copyfileobj(src, out)
    else:
        raise ValueError("Nicht unterstütztes Archivformat.")
    # If archive contains a single folder, descend into it
    entries = [e for e in os.listdir(tmp_dir) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
        return os.path.join(tmp_dir, entries[0])
    return tmp_dir


@app.post("/admin/upload-model")
def admin_upload_model():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    f = request.files.get("model_file")
    if not f:
        return ("Keine Datei hochgeladen", 400)
    filename = f.filename or ""
    filename = os.path.basename(filename)
    allowed = (".zip", ".tar", ".tar.gz", ".tgz", ".pth", ".pth.tar")
    if not any(filename.lower().endswith(ext) for ext in allowed):
        return ("Ungültiger Dateityp. Erlaubt: .zip, .tar, .tar.gz, .pth", 400)

    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    os.makedirs(models_root, exist_ok=True)

    tmp_file = None
    msg = ""
    try:
        # Save to temp
        fd, tmp_file = tempfile.mkstemp(prefix="upload_", suffix=os.path.splitext(filename)[1])
        os.close(fd)
        f.save(tmp_file)

        if filename.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            src_dir = _extract_archive_to_tmp(tmp_file)
            msg = _install_dir_to_models(src_dir, models_root)
        else:
            # Single file upload (likely .pth). Require a config.json alongside? Not available via single file.
            # Place into a default custom folder and inform user to provide config.json.
            target_dir = os.path.join(models_root, "tts_models--custom--uploaded")
            os.makedirs(target_dir, exist_ok=True)
            dst = os.path.join(target_dir, os.path.basename(filename))
            shutil.move(tmp_file, dst)
            tmp_file = None
            msg = f"Datei gespeichert nach {dst}. Bitte sicherstellen, dass config.json vorhanden ist."

        # Reset current model, it will reload next time
        global _tts_instance, _tts_samplerate
        _tts_instance = None
        _tts_samplerate = None
        return redirect(url_for("admin_page", msg=f"{msg}"))
    except Exception as e:
        return redirect(url_for("admin_page", msg=f"Upload-Fehler: {e}"))
    finally:
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass


@app.post("/admin/import-model")
def admin_import_model():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    repo = (request.form.get("repo_url") or "").strip()
    force = bool(request.form.get("force"))
    if not repo:
        return ("Keine Repository-Angabe", 400)

    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    os.makedirs(models_root, exist_ok=True)
    msg = ""
    try:
        if repo.startswith("http://") or repo.startswith("https://"):
            # Fetch archive via HTTP and install
            import requests
            r = requests.get(repo, stream=True, timeout=60)
            r.raise_for_status()
            # Guess filename
            name = os.path.basename(repo.split("?")[0]) or "model.zip"
            fd, tmp_path = tempfile.mkstemp(prefix="import_", suffix=os.path.splitext(name)[1])
            with os.fdopen(fd, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        out.write(chunk)
            src_dir = _extract_archive_to_tmp(tmp_path)
            msg = _install_dir_to_models(src_dir, models_root)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Treat as Coqui model name, optionally skip if exists
            target_name = repo.replace("/", "--")
            target_dir = None
            if target_name.startswith("tts_models--") or target_name.startswith("vocoder_models--"):
                target_dir = os.path.join(models_root, target_name)
            if target_dir and os.path.isdir(target_dir) and not force:
                msg = f"Modell existiert bereits: {target_name} (verwende --force zum Überschreiben)"
            else:
                from TTS.utils.manage import ModelManager
                manager = ModelManager()
                manager.output_prefix = models_root
                manager.download_model(repo)
                msg = f"Modell importiert: {repo}"

        # Reset model
        global _tts_instance, _tts_samplerate
        _tts_instance = None
        _tts_samplerate = None
        return redirect(url_for("admin_page", msg=msg))
    except Exception as e:
        return redirect(url_for("admin_page", msg=f"Import-Fehler: {e}"))


@app.post("/admin/activate-model")
def admin_activate_model():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    selected = (request.form.get("model_name") or "").strip()
    if not selected:
        return redirect(url_for("admin_page", msg="Kein Modell ausgewählt."))

    # Sanitize and validate
    selected = os.path.basename(selected)
    if not (selected.startswith("tts_models--") or selected.startswith("vocoder_models--")):
        return redirect(url_for("admin_page", msg="Ungültiger Modellname."))

    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    abs_path = os.path.normpath(os.path.join(models_root, selected))
    if not abs_path.startswith(models_root) or not os.path.isdir(abs_path):
        return redirect(url_for("admin_page", msg="Modellordner existiert nicht."))

    active_file = os.path.join(base_dir, ".active_model.json")
    data = {"tts": None, "vocoder": None}
    try:
        if os.path.exists(active_file):
            with open(active_file, "r", encoding="utf-8") as f:
                data = json.load(f) or data
    except Exception:
        pass

    if selected.startswith("tts_models--"):
        data["tts"] = selected
    else:
        data["vocoder"] = selected

    try:
        with open(active_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        return redirect(url_for("admin_page", msg=f"Fehler beim Schreiben: {e}"))

    # Reset model so it reloads next call
    global _tts_instance, _tts_samplerate
    with _model_lock:
        _tts_instance = None
        _tts_samplerate = None
    return redirect(url_for("admin_page", msg=f"Modell '{selected}' aktiviert."))


@app.post("/admin/set-active-pair")
def admin_set_active_pair():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    tts_model = (request.form.get("tts_model") or "").strip()
    voc_model = (request.form.get("vocoder_model") or "").strip()

    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")

    def _valid(name: str, prefix: str) -> str:
        name = os.path.basename(name)
        if not name.startswith(prefix):
            return ""
        p = os.path.normpath(os.path.join(models_root, name))
        if not p.startswith(models_root) or not os.path.isdir(p):
            return ""
        return name

    tts_model = _valid(tts_model, "tts_models--")
    voc_model = _valid(voc_model, "vocoder_models--")

    if not tts_model or not voc_model:
        return redirect(url_for("admin_page", msg="Ungültige Auswahl für TTS/Vocoder."))

    active_path = os.path.join(base_dir, ".active_model.json")
    data = {"tts": tts_model, "vocoder": voc_model}
    try:
        with open(active_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        return redirect(url_for("admin_page", msg=f"Fehler beim Schreiben: {e}"))

    global _tts_instance, _tts_samplerate
    with _model_lock:
        _tts_instance = None
        _tts_samplerate = None
    return redirect(url_for("admin_page", msg=f"Aktive Modelle aktualisiert: {tts_model} / {voc_model}"))


@app.post("/admin/delete-model")
def admin_delete_model():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    model_name = os.path.basename(request.form.get("model_name", "").strip())
    if not (model_name.startswith("tts_models--") or model_name.startswith("vocoder_models--")):
        return redirect(url_for("admin_page", msg="Ungültiger Modellname."))
    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    target = os.path.normpath(os.path.join(models_root, model_name))
    if not target.startswith(models_root):
        return redirect(url_for("admin_page", msg="Ungültiger Pfad."))

    # If active, unset
    active_file = os.path.join(base_dir, ".active_model.json")
    try:
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            # Update active selection if it pointed to this
            data = {"tts": None, "vocoder": None}
            if os.path.exists(active_file):
                try:
                    with open(active_file, "r", encoding="utf-8") as f:
                        data = json.load(f) or data
                except Exception:
                    pass
            changed = False
            if data.get("tts") == model_name:
                data["tts"] = None
                changed = True
            if data.get("vocoder") == model_name:
                data["vocoder"] = None
                changed = True
            if changed:
                try:
                    with open(active_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass
            # Reset model (in case the active was removed)
            global _tts_instance, _tts_samplerate
            _tts_instance = None
            _tts_samplerate = None
            msg = f"Modell '{model_name}' gelöscht."
        else:
            msg = "Ordner nicht gefunden."
    except Exception as e:
        msg = f"Fehler beim Löschen: {e}"
    return redirect(url_for("admin_page", msg=msg))


@app.post("/admin/preview")
def admin_preview():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    text = (request.form.get("preview_text") or "").strip()
    if not text:
        return ("Kein Text.", 400)
    try:
        _, mp3_bytes = _synthesize_to_mp3(text)
        return send_file(io.BytesIO(mp3_bytes), mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

application = app 

@app.get("/admin/download-status")
def admin_download_status():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    return jsonify(_download_status)


@app.get("/admin/verify-models")
def admin_verify_models():
    if not _admin_token() or not _is_admin_request():
        return ("Nicht autorisiert", 403)
    base_dir = os.path.dirname(__file__)
    models_root = os.path.join(base_dir, "models")
    results = []
    try:
        for name in sorted(os.listdir(models_root)):
            if not (name.startswith("tts_models--") or name.startswith("vocoder_models--")):
                continue
            path = os.path.join(models_root, name)
            if name.startswith("tts_models--"):
                req = [os.path.join(path, "model_file.pth"), os.path.join(path, "config.json")]
            else:
                req = [os.path.join(path, "model_file.pth.tar"), os.path.join(path, "config.json")]
            missing = [os.path.relpath(p, start=base_dir) for p in req if not os.path.exists(p)]
            results.append({
                "name": name,
                "type": "TTS" if name.startswith("tts_models--") else "Vocoder",
                "path": os.path.relpath(path, start=base_dir),
                "ok": len(missing) == 0,
                "missing": missing,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"items": results})


# ==== Run locally ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

# Optional: avoid favicon 404 noise during local testing
@app.get("/favicon.ico")
def favicon():
    return ("", 204)
