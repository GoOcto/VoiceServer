# 🎙️ Voice Server API

A high-performance Flask-based Text-to-Speech (TTS) server powered by Coqui TTS. This server provides flexible voice synthesis with support for multiple TTS models, voice cloning capabilities, and real-time pitch adjustment for dynamic speech generation.

## ✨ Features

- **Multiple TTS Models**: Support for VITS, Glow-TTS, and XTTS v2 models
- **Voice Cloning**: Clone any voice using the multilingual XTTS v2 model
- **Speed Control**: Real-time speech speed adjustment (0.5x to 2.0x and beyond)
- **RESTful API**: Simple HTTP API for easy integration
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Pre-computed Latents**: Optimized voice loading with speaker embedding caching

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 or 3.11** (Python 3.12 not supported due to dependency constraints)
- **CUDA-compatible GPU** (optional but recommended for better performance)

### Installation

```bash
# Install Python 3.11 (Ubuntu/Debian)
sudo add-apt-repository ppa:deadsnakes/ppa   # if needed
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the voice server
python serve_voice.py
```

The server will start on `http://0.0.0.0:5001` and be accessible from your network.

## 📡 API Usage

### Endpoint: `/api/tts`

**Method**: `POST`  
**Content-Type**: `application/json`

#### Request Body

```json
{
    "utterance": "Your text to synthesize into speech",
    "speed": 1.0
}
```

| Parameter   | Type   | Required | Description                                    |
|------------|--------|----------|------------------------------------------------|
| `utterance` | string | Yes      | Text to convert to speech                      |
| `speed`     | float  | No       | Speech speed multiplier (default: 1.0)        |

#### Response

Returns a WAV audio file that can be played directly or saved.

#### Example with cURL

```bash
curl -X POST http://localhost:5001/api/tts \
  -H "Content-Type: application/json" \
  -d '{"utterance": "Hello, world!", "speed": 1.2}' \
  --output speech.wav
```

## 🎯 Model Configuration

The server supports three TTS models. Configure your preferred model by editing the `DEFAULT_MODEL_TYPE` variable in `serve_voice.py`:

### VITS Model (Default)

```python
DEFAULT_MODEL_TYPE = "vits"
VITS_SPEAKER = "p273"  # Choose from 100+ available speakers
```

### Glow-TTS Model

```python
DEFAULT_MODEL_TYPE = "glow"
```

### XTTS v2 (Voice Cloning)

```python
DEFAULT_MODEL_TYPE = "cloned"
CLONED_VOICE_FILE = "voice_samples/your_voice.wav"
```

## 🎭 Voice Cloning Setup

To use voice cloning with XTTS v2:

1. **Prepare Voice Sample**: Place a clear, 10-30 second WAV file in the `voice_samples/` directory
2. **Update Configuration**: Set the `CLONED_VOICE_FILE` path in `serve_voice.py`
3. **Optional - Pre-compute Latents**: For faster loading, generate speaker embeddings:

```bash
# Edit calc_latents.py to point to your voice sample
python utils/calc_latents.py
```

This creates a `.pth` file with pre-computed speaker embeddings, reducing server startup time.

### Voice Sample Requirements

- **Format**: WAV (16-bit, mono recommended)
- **Duration**: 10-30 seconds for best results
- **Quality**: Clear speech, minimal background noise
- **Language**: English (or target language for multilingual use)

## 🛠️ Development

### Project Structure

```text
VoiceServer/
├── serve_voice.py          # Main Flask server
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── utils/
│   └── calc_latents.py    # Speaker embedding pre-computation
└── voice_samples/
    ├── voice_sample.wav         # Example voice samples
    ├── voice_sample.pth         # Pre-computed embeddings
    └── other_sample.wav         # Additional samples
```

### Customization

- **Port**: Change the port in `app.run(port=5001)`
- **Host**: Modify network accessibility with `host` parameter
- **Models**: Switch between TTS models by changing `DEFAULT_MODEL_TYPE`
- **Speakers**: For VITS, choose from 100+ available speakers

## 📊 Performance Notes

- **GPU**: CUDA acceleration provides ~5-10x faster inference
- **Memory**: XTTS v2 requires ~2-4GB VRAM for optimal performance
- **Latency**: First request may be slower due to model loading
- **Caching**: Pre-computed latents reduce voice cloning startup time

## 🤝 Contributing

This project uses Coqui TTS under the hood. For issues related to TTS models or synthesis quality, please refer to the [Coqui TTS documentation](https://docs.coqui.ai/).

## 📄 License

This project is open source. Please check individual model licenses when using different TTS models.

---

Built with ❤️ using Coqui TTS and Flask
GoOcto
