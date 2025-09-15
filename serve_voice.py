# Coqui TTS API Server
#
# This script provides a Flask-based web server that exposes a single API endpoint
# for text-to-speech (TTS) synthesis using Coqui TTS models. It is designed to be
# simple to configure and use.
#
# --- USAGE ---
#
# See README.md for detailed instructions.
#
# ---
import io
import warnings
from flask import Flask, request, send_file

# Suppress future warnings from libraries
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import numpy as np
from pydub import AudioSegment
from TTS.api import TTS

# --- CONFIGURATION ---
DEFAULT_MODEL_TYPE = "vits"

# Default speed of the generated speech (1.0 is normal, >1.0 is faster, <1.0 is slower)
DEFAULT_SPEED = 1.0

VITS_MODEL = "tts_models/en/vctk/vits"
VITS_SPEAKER = "p273"  # Example speaker from the VCTK dataset

# ones I like: p268, p270, p273   (nope)..p288

# Available speakers: ['ED\n', 'p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p236', 'p237', 'p238', 'p239', 'p240', 'p241', 'p243', 'p244', 'p245', 'p246', 'p247', 'p248', 'p249', 'p250', 'p251', 'p252', 'p253', 'p254', 'p255', 'p256', 'p257', 'p258', 'p259', 'p260', 'p261', 'p262', 'p263', 'p264', 'p265', 'p266', 'p267', 'p268', 'p269', 'p270', 'p271', 'p272', 'p273', 'p274', 'p275', 'p276', 'p277', 'p278', 'p279', 'p280', 'p281', 'p282', 'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', 'p293', 'p294', 'p295', 'p297', 'p298', 'p299', 'p300', 'p301', 'p302', 'p303', 'p304', 'p305', 'p306', 'p307', 'p308', 'p310', 'p311', 'p312', 'p313', 'p314', 'p316', 'p317', 'p318', 'p323', 'p326', 'p329', 'p330', 'p333', 'p334', 'p335', 'p336', 'p339', 'p340', 'p341', 'p343', 'p345', 'p347', 'p351', 'p360', 'p361', 'p362', 'p363', 'p364', 'p374', 'p376']

GLOW_MODEL = "tts_models/en/ljspeech/glow-tts"

CLONED_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
CLONED_VOICE_FILE = "path/to/your/speaker_voice.wav"

# --- END OF CONFIGURATION ---


# --- GLOBAL INITIALIZATION ---
# This section loads the model into memory when the server starts.
print("🚀 Initializing TTS Server...")

# Determine the device to use for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Global variables for the TTS model and any pre-computed data
tts_model = None
xtts_speaker_data = None


def initialize_model():
    """Loads the configured TTS model into memory."""
    global tts_model, xtts_speaker_data

    model_name = None
    if DEFAULT_MODEL_TYPE == 'vits':
        model_name = VITS_MODEL
    elif DEFAULT_MODEL_TYPE == 'glow':
        model_name = GLOW_MODEL
    elif DEFAULT_MODEL_TYPE == 'cloned':
        model_name = CLONED_MODEL
    else:
        raise ValueError(f"Invalid DEFAULT_MODEL_TYPE: '{DEFAULT_MODEL_TYPE}'. Must be 'vits', 'glow', or 'cloned'.")

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsArgs
    from TTS.tts.models.xtts import XttsAudioConfig
    torch.serialization.add_safe_globals([BaseDatasetConfig, XttsConfig, XttsArgs, XttsAudioConfig])

    print(f". Loading model: {model_name}...", end="", flush=True)
    tts_model = TTS(model_name=model_name).to(device)
    print(f"\r✅ Model loaded: {model_name}...")

    # Pre-computation for specific models
    if DEFAULT_MODEL_TYPE == 'vits':
        if VITS_SPEAKER not in tts_model.speakers:
            print(f"⚠️ Warning: Default speaker '{VITS_SPEAKER}' not found.")
            print(f"Available speakers: {tts_model.speakers}")
            raise ValueError(f"Default speaker '{VITS_SPEAKER}' not found in model.")
        print(f"   Default speaker set to: {VITS_SPEAKER}")

    elif DEFAULT_MODEL_TYPE == 'cloned':
        print("   Computing speaker latents for XTTS model...")
        try:
            xtts_speaker_data = {"speaker_wav": CLONED_VOICE_FILE}
            print("   ✅ Speaker latents computed and cached.")
        except FileNotFoundError:
            print(f"❌ Error: The audio file for the cloned voice was not found at '{CLONED_VOICE_FILE}'.")
            print("   Please update the CLONED_VOICE_FILE path in the script.")
            raise
        except Exception as e:
            print(f"❌ An error occurred during speaker latent computation: {e}")
            raise


# Initialize the model on startup
initialize_model()

# --- FLASK API SERVER ---
app = Flask(__name__)

@app.route('/api/tts', methods=['POST'])
def api_tts():
    """
    The main API endpoint for TTS generation.
    Expects a JSON body with an "utterance" key and an optional "speed" key.
    """
    # Validate request
    if not request.is_json:
        return "Invalid request: Content-Type must be application/json", 415

    data = request.get_json()
    if not data or 'utterance' not in data:
        return "Invalid request: JSON body must contain an 'utterance' key.", 400

    utterance = data['utterance']
    # Get speed from request, or use the configured default
    speed = float(data.get('speed', DEFAULT_SPEED))
    
    print(f"🎤 Received request for utterance: \"{utterance[:100]}...\" (Speed: {speed}x)")

    waveform = None
    try:
        # Generate audio based on the configured model type
        if DEFAULT_MODEL_TYPE == 'vits':
            waveform = tts_model.tts(text=utterance, speaker=VITS_SPEAKER)
        elif DEFAULT_MODEL_TYPE == 'glow':
            waveform = tts_model.tts(text=utterance)
        elif DEFAULT_MODEL_TYPE == 'cloned':
            waveform = tts_model.tts(
                text=utterance,
                language="en",
                speaker_wav=xtts_speaker_data['speaker_wav']
            )

    except Exception as e:
        print(f"❌ Error during TTS generation: {e}")
        return "An error occurred during audio generation.", 500

    if waveform is None:
        return "Audio generation failed for an unknown reason.", 500

    sample_rate = tts_model.synthesizer.output_sample_rate
    
    # Convert float waveform to 16-bit PCM
    waveform_int16 = (np.array(waveform) * 32767).astype(np.int16)
    
    # Create a pydub AudioSegment
    audio = AudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=waveform_int16.dtype.itemsize,
        channels=1
    )

    # Apply speed adjustment if necessary
    if speed != 1.0:
        altered_audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * speed)
        })
    else:
        altered_audio = audio

    # Export the final audio to an in-memory binary buffer
    wav_buffer = io.BytesIO()
    altered_audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0) # Rewind the buffer to the beginning

    print("✅ Audio generated successfully. Sending response.")

    # Send the WAV file as the response
    return send_file(
        wav_buffer,
        mimetype='audio/wav',
        as_attachment=False, # Use 'True' to force download
        download_name='output.wav'
    )

if __name__ == '__main__':
    # Run the Flask app
    # Use host='0.0.0.0' to make the server accessible from other devices on your network
    app.run(host='0.0.0.0', port=5001)

