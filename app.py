import os
import io
import time
from uuid import uuid4
from typing import Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)


# Environment configuration
DEFAULT_MODEL_NAME = "tts_models/de/thorsten/tacotron2-DDC"
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", DEFAULT_MODEL_NAME)
USE_S3 = os.getenv("USE_S3", "true").lower() in {"1", "true", "yes"}
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "plattdeutsch-tts-audio")


_tts_instance = None  # Lazy-loaded Coqui TTS model
_tts_samplerate: Optional[int] = None
_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _load_tts() -> Tuple[object, int]:
    """Load and cache the TTS model. Returns (tts_instance, sample_rate)."""
    global _tts_instance, _tts_samplerate
    if _tts_instance is not None and _tts_samplerate is not None:
        return _tts_instance, _tts_samplerate

    # Import here to avoid import cost for simple health checks
    from TTS.api import TTS  # type: ignore

    # Auto-select CPU; GPU selection can be added later if needed
    tts = TTS(model_name=TTS_MODEL_NAME)

    # Try to determine the model output sample rate robustly across TTS versions
    sr = None
    try:
        sr = getattr(tts, "output_sample_rate", None)
    except Exception:
        sr = None
    if not sr:
        try:
            sr = getattr(tts.synthesizer, "output_sample_rate", None)
        except Exception:
            sr = None
    if not sr:
        try:
            sr = getattr(getattr(tts, "synthesizer", None), "tts_config", None)
            sr = getattr(getattr(sr, "audio", None), "sample_rate", None)
        except Exception:
            sr = None
    if not sr:
        sr = 22050  # Reasonable default for Tacotron2 models

    _tts_instance = tts
    _tts_samplerate = int(sr)
    return _tts_instance, _tts_samplerate


def _wav_to_mp3_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    import lameenc

    # Ensure mono int16 PCM little-endian
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16).tobytes()

    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(int(sample_rate))
    enc.set_channels(1)
    enc.set_quality(2)
    mp3_data = enc.encode(pcm16)
    mp3_data += enc.flush()
    return mp3_data


def _synthesize_to_mp3(text: str) -> Tuple[str, bytes]:
    tts, sr = _load_tts()

    # Generate waveform (float32 numpy array)
    # Coqui TTS API returns a numpy array from tts.tts
    # The German Thorsten model is single-speaker; language/speaker args not needed
    audio = tts.tts(text)
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    mp3_bytes = _wav_to_mp3_bytes(audio, sr)

    file_id = uuid4().hex[:8]
    filename = f"tts_{int(time.time())}_{file_id}.mp3"
    return filename, mp3_bytes


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/synthesize")
def synthesize():
    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Bitte Text eingeben."}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text ist zu lang (max. 1000 Zeichen)."}), 400

    try:
        filename, mp3_bytes = _synthesize_to_mp3(text)
    except Exception as e:
        return jsonify({"error": f"TTS-Fehler: {e}"}), 500

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

    # Fallback: return file directly
    return send_file(
        io.BytesIO(mp3_bytes),
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name=filename,
    )


@app.get("/health")
def health():
    loaded = _tts_instance is not None
    sample_rate = _tts_samplerate if loaded else None
    return jsonify(
        {
            "status": "ok",
            "model_name": TTS_MODEL_NAME,
            "model_loaded": bool(loaded),
            "sample_rate": sample_rate,
            "use_s3": bool(USE_S3),
            "bucket": S3_BUCKET_NAME if USE_S3 else None,
            "region": AWS_REGION,
        }
    )


if __name__ == "__main__":
    # Local dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

