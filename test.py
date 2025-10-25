from TTS.api import TTS

# Load the German model
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC")

# Generate and save audio
tts.tts_to_file(text="Hallo Welt,mein Name ist Klaus, dies ist ein Test mit Plattdeutsch TTS.", file_path="output.wav")
print(" Audio file saved as output.wav")
