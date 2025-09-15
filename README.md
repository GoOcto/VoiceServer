# Voice Server API

Uses Python TTS library with switches to allow different models to be used (vits, glow and cloned). Runs as an API server. I also allows for speed control to make the voice sound more comical.

## Set Up

Python 3.10 or 3.11 is recommended. (Python 3.12 is not supported due to dependency issues.)

```bash

# install python3.11
sudo add-apt-repository ppa:deadsnakes/ppa   # may be required if the next line fails
sudo apt install python3.11 python3.11-venv python3.11-dev

# make your venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the server from within your venv.

```bash
python serve_voice.py
```

It will serve up an API on the port specified in the code. You can use Postman, for example, to test. Send JSON data as in the following example:

```json
{
    "utterance": "Tell me a story.",
    "speed": 1.2
}
```

## The cloned model

To use the cloned model you will need to provide voice samples in the form of a wav file. The file name can be found in the code.

```
CLONED_VOICE_FILE = "path/to/your/speaker_voice.wav"
```

Additionally, to help optimize load time, you can use the utility `calc_latents.py` (in the utils folder) to pregenerate a much smaller .pth file which you can then load directly with the line shown above (instead of the .wav file). 
