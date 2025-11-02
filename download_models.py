from TTS.utils.manage import ModelManager
import os

# Path where models will be stored
out_path = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out_path, exist_ok=True)

# Create ModelManager and set custom output folder
manager = ModelManager()
manager.output_prefix = out_path  # 👈 redirect cache/download path

# Model name
model_name = "tts_models/de/thorsten/tacotron2-DDC"
vocoder_name = "vocoder_models/de/thorsten/hifigan_v1"

# Download model + vocoder into /models
manager.download_model(model_name)
manager.download_model(vocoder_name)




print(f"✅ Model '{model_name}' downloaded into {out_path}")
print(f"✅ Vocoder '{vocoder_name}' downloaded into {out_path}")
