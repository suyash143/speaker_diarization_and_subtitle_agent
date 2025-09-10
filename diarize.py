import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

AUDIO_FILE = "inputaudio.wav"

# Load environment variables from .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

try:
    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    if pipeline is None:
        raise RuntimeError("Pipeline could not be loaded. Check your Hugging Face token and permissions. Make sure you have accepted user conditions for all required models at https://hf.co/pyannote/speaker-diarization and https://hf.co/pyannote/segmentation.")
    print("Pipeline loaded. Running diarization...")
    diarization = pipeline(AUDIO_FILE)
    segments = list(diarization.itertracks(yield_label=True))

    for turn, _, speaker in segments:
        print(f"{turn.start:.2f} - {turn.end:.2f}: Speaker {speaker}")

    with open("diarization_segments.txt", "w") as f:
        for turn, _, speaker in segments:
            f.write(f"{turn.start:.2f},{turn.end:.2f},Speaker {speaker}\n")
except Exception as e:
    print(f"Error during diarization: {e}")

