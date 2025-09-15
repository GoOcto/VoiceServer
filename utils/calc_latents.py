# This code should be run only ONCE for each new voice
import torch
from TTS.api import TTS

# TODO: make the main CoquiTTS file save the latents automatically
# and use them if they've already been generated

# Add the specific configuration class to the safe list
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([
    XttsConfig, 
    Xtts, 
    XttsArgs, 
    XttsAudioConfig, 
    BaseDatasetConfig])


device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_wav_path = "voice_samples/aurora_long.wav"

print("Loading model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print(f"Computing speaker latents from {speaker_wav_path}...")

# This is the core function to extract the voice characteristics
gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=speaker_wav_path)

# Define a path for your saved embedding
embedding_path = "voice_samples/aurora_long.pth"

print(f"Saving speaker embedding to {embedding_path}...")
# Save the tensors in a dictionary
torch.save({
    "gpt_cond_latent": gpt_cond_latent,
    "speaker_embedding": speaker_embedding
}, embedding_path)

print("Done.")