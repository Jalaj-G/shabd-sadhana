import gradio as gr
import os, sys
import pandas as pd
from utils.helpers import get_audio_transcript_pairs, extract_zip
import time, subprocess
from pathlib import Path
from utils.logger import setup_logger
logger = setup_logger()


model_state = {"selected_model": "openai/whisper-base"}
dataset_dir = {"audio_dir": "data/audios", "transcript_dir": "data/transcripts"}

# Block 1: Model selection
def handle_model_selection(choice, uploaded_file):
    if choice != "Upload custom model":
        model_state["selected_model"] = choice
        return f"Selected pre-trained model: {choice}"
    elif uploaded_file:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", uploaded_file.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_file.read())
        model_state["selected_model"] = model_path
        return f"Custom model uploaded: {uploaded_file.name}"
    else:
        return "Please upload a model file."

def build_model_selector_block():
    with gr.Group():
        gr.Markdown("### 🔧 Model Selection")

        model_choice = gr.Dropdown(
            ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large", "Upload custom model"],
            label="Select Model",
            value="openai/whisper-base"
        )
        model_upload = gr.File(label="Upload custom model (.pt/.bin)", file_types=[".pt", ".bin"])
        model_status = gr.Textbox(label="Model Status", interactive=False)

        model_choice.change(fn=handle_model_selection, inputs=[model_choice, model_upload], outputs=model_status)
        model_upload.change(fn=handle_model_selection, inputs=[model_choice, model_upload], outputs=model_status)

    return model_choice, model_upload, model_status

# Block 2: Dataset upload
def handle_zip_upload(zip_file):
    if zip_file is None:
        return pd.DataFrame(), "❌ No file uploaded."

    os.makedirs("data", exist_ok=True)
    extract_zip(zip_file.name, "data")
    matched = get_audio_transcript_pairs("data")

    if not matched:
        return pd.DataFrame(), "❌ No valid audio-transcript pairs found."

    df = pd.DataFrame(matched)

    dataset_dir["audio_dir"] = df[0].apply(lambda x: os.path.dirname(x)).unique()[0]
    dataset_dir["transcript_dir"] = df[1].apply(lambda x: os.path.dirname(x)).unique()[0]
    logger.info(f"Audio directory: {dataset_dir['audio_dir']}")
    logger.info(f"Transcript directory: {dataset_dir['transcript_dir']}")

    return df, "✅ Valid audio-transcript pairs found."

def build_upload_block():
    with gr.Group():
        gr.Markdown("### 📂 Upload Dataset (.zip with `audios/` and optional `transcripts/`)")

        zip_upload = gr.File(label="Upload ZIP file", file_types=[".zip"])
        upload_result = gr.Textbox(label="Upload Status", interactive=False)
        file_preview = gr.Dataframe(label="Matched Files (Preview)")

        zip_upload.change(fn=handle_zip_upload, inputs=zip_upload, outputs=[file_preview, upload_result])

    return zip_upload, file_preview, upload_result


def handle_finetune(epoch, batch, lr, finetune_mode):
    model_name_or_path = model_state.get("selected_model")
    audio_dir = dataset_dir["audio_dir"]
    transcript_dir = dataset_dir["transcript_dir"]

    if not model_name_or_path:
        return "❌ No model selected. Please choose a pre-trained model or upload one."

    logger.info(f"Model path being passed: {model_name_or_path}")

    # Absolute path to scripts/train.py (safer than relying on CWD)
    train_script = Path(__file__).with_name("scripts") / "train.py"

    try:
        subprocess.run(
            [
                sys.executable,                # ← THIS is the venv’s interpreter
                train_script,
                "--model_name_or_path", model_name_or_path,
                "--audio_dir", audio_dir,
                "--transcript_dir", transcript_dir,
                "--output_dir", "output",
                "--epochs", str(epoch),
                "--per_device_train_batch_size", str(batch),
                "--learning_rate", str(lr),
                "--finetune_mode", finetune_mode,
            ],
            check=True,
            cwd=Path(__file__).parent,        # keeps relative imports happy
        )
        logger.info("Training finished. Model saved in /output/final_model")

        zip_path = os.path.abspath("output/final_model.zip")
        return zip_path if os.path.exists(zip_path) else None

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return None


def build_finetune_block():
    with gr.Group():
        gr.Markdown("### ⚙️ Fine-Tuning Configuration")

        with gr.Row():
            epochs = gr.Number(value=3, label="Epochs")
            batch_size = gr.Number(value=8, label="Batch Size")
            learning_rate = gr.Number(value=1e-5, label="Learning Rate")
            finetune_mode = gr.Dropdown(
                choices=["full", "lora", "qlora"],
                value="full",
                label="Fine-Tuning Mode"
            )

        start_button = gr.Button("🚀 Start Fine-Tuning")
        # status_output = gr.Textbox(label="Training Status", interactive=False)
        model_download = gr.File(label="Download Fine-Tuned Model", interactive=False)

        start_button.click(
            fn=handle_finetune,
            inputs=[epochs, batch_size, learning_rate, finetune_mode],
            outputs=model_download
        )

    return epochs, batch_size, learning_rate, start_button, model_download



# Assemble full UI
def launch_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🕉️ Shabd-Sādhana – Train Your Own STT Model")

        build_model_selector_block()
        build_upload_block()
        build_finetune_block()

    demo.launch()

# Run app
if __name__ == "__main__":
    launch_ui()
