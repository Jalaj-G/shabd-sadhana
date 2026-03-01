import os
import sys
import shutil
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch.nn as nn

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root
from scripts.prepare_data import build_dataset
from utils.logger import setup_logger
from utils.metrics import compute_wer

logger = setup_logger()

# ------------------------------
# 🔧 ARGUMENT PARSING
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--audio_dir", type=str, required=True)
parser.add_argument("--transcript_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--finetune_mode", type=str, default="full", choices=["full", "lora", "qlora"])
args = parser.parse_args()

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
            # padding=self.padding,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            # padding=self.padding,
            return_tensors="pt"
        )

        # Mask padding token id in labels
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ─── tiny wrapper that neutralises `input_ids` ────────────────────────────────
class WhisperPeftWrapper(nn.Module):
    """
    Thin shim around a PEFT-wrapped Whisper model.

    • Discards any accidental `input_ids`
    • Calls  self.model.base_model(...)  directly, so PeftModel can’t
      re-insert `input_ids`.
    • Proxies .config and .generate for full Trainer compatibility
    """
    def __init__(self, peft_model):
        super().__init__()
        self.model = peft_model            # the PEFT model (LoRA / QLoRA)

    def forward(self, *args, **kwargs):
        kwargs.pop("input_ids", None)      # get rid of it (if present)
        kwargs.pop("num_items_in_batch", None)  # <- THIS is the fix
        return self.model.base_model(*args, **kwargs)   # call Whisper itself

    # --- proxy helpers -------------------------------------------------------
    @property
    def config(self):
        return self.model.config

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        return self.model.save_pretrained(save_directory, **kwargs)



# ------------------------------
# 📦 LOAD DATASET
# ------------------------------
logger.info("Loading dataset...")
dataset_dict = build_dataset(args.audio_dir, args.transcript_dir, test_size=0.1)

# ------------------------------
# 🧠 LOAD MODEL + PROCESSOR
# ------------------------------
processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="en", task="transcribe")

load_kwargs = {}
if args.finetune_mode == "qlora":
    load_kwargs["load_in_4bit"] = True
    load_kwargs["device_map"] = "auto"
elif args.finetune_mode == "lora":
    load_kwargs["device_map"] = "auto"

model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)

# Avoid forcing/suppressing tokens
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# ------------------------------
# 🪄 PEFT SETUP
# ------------------------------
if args.finetune_mode in ["lora", "qlora"]:
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg, low_cpu_mem_usage=False)
    model.print_trainable_parameters()

    # 🔑 wrap only for LoRA / QLoRA
    model = WhisperPeftWrapper(model)

# ------------------------------
# ✂️ TOKENIZATION
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

# ------------------------------
# 📊 EVALUATION METRICS
# ------------------------------
def compute_metrics(pred):
    """
    Compute WER metric during evaluation.

    Called by Trainer after each evaluation run.
    """
    import numpy as np

    # Extract predictions and labels
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in labels (used for padding) with pad_token_id
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    # Decode predictions and labels to text
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer_result = compute_wer(references=label_str, predictions=pred_str)

    return wer_result

logger.info("Tokenizing dataset...")
processed_train = dataset_dict["train"].map(preprocess)
processed_eval = dataset_dict["eval"].map(preprocess)
# print(processed_train[0])

# ------------------------------
# ⚙️ TRAINING SETUP
# ------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=1,
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    warmup_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_strategy="steps",
    logging_steps=1,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="none",
    fp16=False,
    bf16=False,
    predict_with_generate=True,
    generation_max_length=225,
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# logger.info(f"data collator: {data_collator}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ------------------------------
# 🚀 START TRAINING
# ------------------------------
model.config.use_cache = False
logger.info("Fine-tuning started...")
trainer.train()

# ------------------------------
# 💾 SAVE MODEL
# ------------------------------
# Load best checkpoint (has lowest WER)
best_checkpoint_path = trainer.state.best_model_checkpoint
if best_checkpoint_path:
    logger.info(f"Best checkpoint found at: {best_checkpoint_path}")
    if args.finetune_mode in ["lora", "qlora"]:
        logger.info(f"Loading best checkpoint for PEFT model...")
        from peft import PeftModel
        base_model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.model = PeftModel.from_pretrained(base_model, best_checkpoint_path)

if args.finetune_mode in ["lora", "qlora"]:
    model.model = model.model.merge_and_unload()  # merge LoRA into base weights

final_model_path = os.path.join(args.output_dir, "final_model")
os.makedirs(final_model_path, exist_ok=True)
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