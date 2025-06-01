import os
import srt
import torchaudio
from datasets import Dataset
from pathlib import Path
import numpy as np
import subprocess
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root

from utils.logger import setup_logger
logger = setup_logger()


torchaudio.set_audio_backend("soundfile")
logger.info(f"Torchaudio backend: {torchaudio.get_audio_backend()}")

def parse_srt_file(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        subtitles = list(srt.parse(content))
        return [{
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
            "text": sub.content.replace("\n", " ")
        } for sub in subtitles]

def load_audio(audio_path):
    ext = os.path.splitext(audio_path)[1].lower()
    
    # If input is mp3, convert to wav using ffmpeg
    if ext == ".mp3":
        wav_path = audio_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):  # avoid re-conversion
            logger.info(f"Converting MP3 to WAV: {audio_path} → {wav_path}")
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-acodec", "pcm_s16le", "-ar", "16000", wav_path  # mono 16-bit PCM at 16kHz
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"WAV conversion failed: {wav_path} not found")
            logger.info(f"Converted WAV ready: {wav_path}")
        audio_path = wav_path

    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform[0].numpy(), sample_rate

def build_dataset(audio_dir, transcript_dir):
    data = []

    for fname in os.listdir(audio_dir):
        logger.info(f"Found audio file: {fname}")

        stem, ext = os.path.splitext(fname)
        if ext.lower() not in [".wav", ".mp3"]:
            continue

        audio_path = os.path.join(audio_dir, fname)
        srt_path = os.path.join(transcript_dir, stem + ".srt")

        if not os.path.exists(srt_path):
            logger.warning(f"No transcript found for {fname}")
            continue

        segments = parse_srt_file(srt_path)
        logger.info(f"Parsed {len(segments)} segments from {srt_path}")
        audio_array, sr = load_audio(audio_path)

        for seg in segments:
            start_sample = int(seg["start"] * sr)
            end_sample = min(int(seg["end"] * sr), len(audio_array))

            # if end_sample <= start_sample or end_sample > len(audio_array):
            #     logger.warning(f"Skipping invalid segment: start={start_sample}, end={end_sample}")
            #     continue

            segment_audio = audio_array[start_sample:end_sample]

            data.append({
                "audio": {
                    "array": segment_audio,
                    "sampling_rate": sr
                },
                "text": seg["text"]
            })
    # This produces a datasets.Dataset object, which we can pass directly to HuggingFace Trainer.
    return Dataset.from_list(data)
