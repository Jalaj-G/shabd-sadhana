# 🕉️ Shabd-Sādhana – Fine-Tune Your Own Speech-to-Text (STT) Model

**Shabd-Sādhana** is an interactive tool to fine-tune [OpenAI Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) models on your custom speech and transcript data. It features a clean [Gradio](https://gradio.app/) UI that allows you to upload data, configure training, and download your own STT model — all in a few clicks.

---

## 🚀 Features

- 🎙️ **Model Selection:** Use pre-trained Whisper models or upload your own.
- 📁 **Dataset Upload:** Upload a ZIP containing audio (`.mp3`/`.wav`) and matching transcript (`.srt`) files.
- ⚙️ **Interactive Training:** Configure training parameters like epochs and batch size via the UI.
- 📦 **Model Download:** Export and download the fine-tuned model as a ZIP file.

---

## 🗂️ Project Structure

```
shabda_sadhana/
├── app.py                # Gradio interface & orchestration
├── data/                 # Uploaded and preprocessed dataset
│   ├── audios/
│   └── transcripts/
├── models/               # (Optional) user-uploaded Whisper models
├── output/
│   ├── final_model/      # Saved model directory
│   └── final_model.zip   # Zipped model for download
├── scripts/
│   ├── prepare_data.py   # Audio + transcript parsing
│   └── train.py          # Whisper fine-tuning logic
├── utils/
│   └── helpers.py        # Utility functions
└── requirements.txt
```

## ⚡ Quickstart

### 1. Clone & Activate Virtual Environment

```sh
git clone <your-repo-url>
cd shabda_sadhana

# Set up and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2. Install Python Dependencies

```sh
pip install -r requirements.txt
```

### 3. Install FFmpeg

**Windows:**
- Download the latest FFmpeg build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html).
- Extract the ZIP and add the `bin` folder to your system `PATH`.

**macOS (with Homebrew):**
```sh
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```sh
sudo apt-get update
sudo apt-get install ffmpeg
```

---
### 4. **Prepare Your Dataset**

- Prepare a ZIP file containing:
  - An `audios/` folder with `.wav` or `.mp3` files.
  - A `transcripts/` folder with matching `.srt` or `.txt` files (same stem as audio).

### 5. **Launch the App**

```sh
python app.py
```

- Open the Gradio UI in your browser. Gradio usually opens at `http://localhost:7860` by default.
- Select or upload a model and confirm the Model Status on UI.
- Upload your dataset ZIP.
- Configure training parameters and start fine-tuning.
- Download the resulting model ZIP.

---

## 🧩 Key Components

- [`app.py`](app.py): Gradio web interface and workflow orchestration.
- [`scripts/prepare_data.py`](scripts/prepare_data.py): Audio/transcript parsing and dataset preparation.
- [`scripts/train.py`](scripts/train.py): Fine-tuning pipeline using HuggingFace Transformers.
- [`utils/helpers.py`](utils/helpers.py): Utilities for ZIP extraction and file pairing.

---

## 📝 Notes

- Audio and transcript files must have matching filenames (e.g., `audio1.wav` ↔ `audio1.srt`).
- The fine-tuned model is saved in `output/final_model.zip`.
- For best results, ensure transcripts are accurate and time-aligned.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Gradio](https://gradio.app/)

