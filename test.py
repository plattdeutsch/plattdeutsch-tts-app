import os
import torch
import collections
from TTS.utils.radam import RAdam
from TTS.api import TTS

# ✅ Allow Coqui’s and Python’s internal classes for model loading (PyTorch ≥ 2.6)
torch.serialization.add_safe_globals([RAdam, collections.defaultdict])
torch.serialization.add_safe_globals([RAdam, collections.defaultdict, dict])

# === Model paths (offline, local) ===
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "tts_models--de--thorsten--tacotron2-DDA")
VOCODER_DIR = os.path.join(BASE_DIR, "models", "vocoder_models--en--ljspeech--hifigan_v2")

# === Initialize Coqui TTS with local preloaded models ===
tts = TTS(
    model_path=os.path.join(MODEL_DIR, "model.pth"),
    config_path=os.path.join(MODEL_DIR, "config.json"),
    vocoder_path=os.path.join(VOCODER_DIR, "model.pth"),
    vocoder_config_path=os.path.join(VOCODER_DIR, "config.json")
)


# === Generate and save audio ===
tts.tts_to_file(
    text="Hallo Welt, mein Name ist Klaus. Dies ist ein Test mit Plattdeutsch TTS.",
    file_path="output.wav"
)

print("✅ Audio file saved as output.wav")
