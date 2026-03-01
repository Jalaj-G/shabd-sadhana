# scripts/infer.py

import os
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio.transforms as T

def load_model_and_processor(model_dir):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return processor, model


def transcribe_audio(audio_path, model_dir):
    processor, model = load_model_and_processor(model_dir)

    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    input_features = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription
