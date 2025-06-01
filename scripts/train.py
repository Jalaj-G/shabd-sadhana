import os
import argparse
import sys, pprint
# print("PATH RIGHT BEFORE torch import:\n", pprint.pformat(sys.path[:8]))
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from prepare_data import build_dataset
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root

from utils.logger import setup_logger
logger = setup_logger()

# ------------------------------
# 🔧 ARGUMENT PARSING
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model name or path to local model directory")
parser.add_argument("--audio_dir", type=str, required=True)
parser.add_argument("--transcript_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save fine-tuned model")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
args = parser.parse_args()

# ------------------------------
# 📁 PATH SETUP
# ------------------------------
AUDIO_DIR = args.audio_dir
TRANSCRIPT_DIR = args.transcript_dir

# ------------------------------
# 🗂 DATA COLLATOR
# ------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch


# ------------------------------
# 📦 LOAD DATASET
# ------------------------------
logger.info("Loading dataset...")
dataset = build_dataset(AUDIO_DIR, TRANSCRIPT_DIR)

# - -----------------------------
# 🧠 LOAD MODEL + PROCESSOR
# ------------------------------
def resolve_model(model_path_or_name):
    if os.path.exists(model_path_or_name):
        logger.info(f"Loading model from local path: {model_path_or_name}")
        return model_path_or_name
    else:
        logger.info(f"Downloading model: {model_path_or_name}")
        return model_path_or_name  # HuggingFace will download & cache

model_source = resolve_model(args.model_name_or_path)

processor = WhisperProcessor.from_pretrained(model_source, language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    model_source,
    # load_in_8bit=True,
    )

# ------------------------------
# ✂️ TOKENIZE DATA
# ------------------------------
def preprocess(batch):
    # Convert audio to features
    inputs = processor(
        batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"],
        return_tensors="pt"
    )

    input_features = inputs.input_features[0]

    # Convert text to labels
    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="longest",
        truncation=True
    ).input_ids[0]

    return {
        "input_features": input_features,
        "labels": labels
    }

logger.info("Tokenizing dataset...")
processed_dataset = dataset.map(preprocess)

# ------------------------------
# ⚙️ TRAINING SETUP
# ------------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=2,
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    # save_steps=100,
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="no",
    save_total_limit=2,
    eval_strategy="no",
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    use_cpu=True,
    remove_unused_columns=False,
    report_to="none",  # Disable reporting to avoid issues in non-interactive environments
)

logger.info(f"Dataset size: {len(processed_dataset)}")
logger.info(f"Processed dataset: {processed_dataset}")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)

# ------------------------------
# 🚀 START TRAINING
# ------------------------------
logger.info("Fine-tuning started...")
trainer.train()

# ------------------------------
# 💾 SAVE FINAL MODEL
# ------------------------------
final_model_path = os.path.join(args.output_dir, "final_model")
model.save_pretrained(final_model_path)
processor.save_pretrained(final_model_path)
logger.info(f"Fine-tuning complete. Model saved at: {final_model_path}")

# ------------------------------
# ZIP MODEL
# ------------------------------

zip_path = final_model_path + ".zip"

# Remove zip if it exists
if os.path.exists(zip_path):
    os.remove(zip_path)

# Create zip
shutil.make_archive(final_model_path, 'zip', final_model_path)
logger.info(f"Zipped model saved at: {zip_path}")