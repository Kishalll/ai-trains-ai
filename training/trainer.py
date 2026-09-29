from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Optional

import deps
from training.dataset import load_all_training_data

MODEL_MAP = {
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def get_base_model_id(student_model: str) -> str:
    cleaned = student_model.strip().lower()
    return MODEL_MAP.get(cleaned, cleaned)


def detect_hardware():
    import torch
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "is_gpu": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
    return {
        "device": "cpu",
        "is_gpu": False,
        "torch_dtype": torch.float32,
        "device_map": None,
    }


def format_dataset_for_sft(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for ex in examples:
        formatted.append(
            {
                "messages": [
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ]
            }
        )
    return formatted


def train_role(
    role_path: Path,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_steps: Optional[int] = None,
    gradient_accumulation_steps: int = 4,
) -> dict[str, Any]:
    if not deps.require_group("training"):
        raise RuntimeError("Training dependencies (torch, transformers, peft, trl, datasets) are required.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    role_path = Path(role_path)
    cfg_file = role_path / "role.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Role config not found at {cfg_file}")

    import yaml
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    student_model = cfg.get("student_model", "qwen2.5:1.5b")
    base_model_id = get_base_model_id(student_model)

    raw_examples = load_all_training_data(role_path)
    if not raw_examples:
        raise ValueError(f"No training data found in {role_path / 'training_data'}. Run import-training first.")

    sft_data = format_dataset_for_sft(raw_examples)
    train_dataset = Dataset.from_list(sft_data)

    hw = detect_hardware()
    output_model_dir = role_path / "model"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for '{base_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model '{base_model_id}' on {hw['device'].upper()}...")
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": hw["torch_dtype"],
    }
    if hw["device_map"]:
        model_kwargs["device_map"] = hw["device_map"]

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    steps_per_epoch = max(1, len(sft_data) // (batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    training_args = TrainingArguments(
        output_dir=str(output_model_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=not hw["is_gpu"],
        max_steps=max_steps if max_steps is not None else -1,
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print(f"Starting fine-tuning: {len(raw_examples)} examples, {epochs} epoch(s)...")
    start_time = datetime.now()
    train_result = trainer.train()
    duration_secs = (datetime.now() - start_time).total_seconds()

    print(f"Saving LoRA adapters to {output_model_dir}...")
    trainer.model.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))

    # Save training status
    status = {
        "status": "completed",
        "role": role_path.name,
        "student_model": student_model,
        "base_model": base_model_id,
        "examples_count": len(raw_examples),
        "epochs": epochs,
        "training_loss": round(float(train_result.training_loss), 4),
        "duration_seconds": round(duration_secs, 1),
        "device": hw["device"],
        "completed_at": datetime.now().isoformat(),
    }

    with open(output_model_dir / "training_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    return status


def get_training_status(role_path: Path) -> Optional[dict[str, Any]]:
    status_file = Path(role_path) / "model" / "training_status.json"
    if not status_file.exists():
        return None
    with open(status_file, "r", encoding="utf-8") as f:
        return json.load(f)
