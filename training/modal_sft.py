"""
Modal SFT (Supervised Fine-Tuning) for Probability Tree Models

This script implements Phase 1 training: learning to predict events that actually happened.
Uses LoRA rank 64 for sufficient learning capacity.
"""

import modal
import json
from pathlib import Path

# Modal app
app = modal.App("psychohistory-sft")

# Shared volume for data and models
volume = modal.Volume.from_name("psychohistory-data", create_if_missing=True)

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "wandb>=0.16.0",
        "trl>=0.7.0",
    )
)


@app.function(
    image=image,
    gpu="H100",  # H100 GPU (80GB) for GPT OSS 20B
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_sft_impl(
    model_name: str = "openai/gpt-oss-20b",  # OpenAI's GPT OSS 20B
    data_path: str = "/data/synthetic_cases.jsonl",
    output_dir: str = "/data/models/sft",
    lora_rank: int = 64,
    learning_rate: float = 3e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
):
    """
    Train SFT model with LoRA rank 64

    Args:
        model_name: Base model to fine-tune
        data_path: Path to training data (JSONL)
        output_dir: Where to save checkpoints
        lora_rank: LoRA rank (64 for SFT)
        learning_rate: Learning rate (10x higher than full FT)
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
    """
    import torch
    import os
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import wandb

    # Handle HuggingFace token (Modal uses HF_TOKEN)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        print("✅ HuggingFace token configured")
    else:
        print("⚠️  No HuggingFace token found")

    print(f"🚀 Starting SFT training")
    print(f"  Model: {model_name}")
    print(f"  LoRA Rank: {lora_rank}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Data: {data_path}")

    # Initialize wandb (optional)
    try:
        wandb.init(
            project="psychohistory-training",
            name=f"sft-rank{lora_rank}",
            config={
                "model": model_name,
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "epochs": num_epochs,
            }
        )
    except:
        print("⚠️  WandB not configured, skipping logging")

    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("\n📥 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA (Thinking Machines recommendations)
    print(f"\n🔧 Configuring LoRA with rank {lora_rank}...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,  # Alpha = 2*rank
        target_modules="all-linear",  # Apply to ALL layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print(f"\n📚 Loading training data from {data_path}...")
    cases = []
    with open(data_path) as f:
        for line in f:
            cases.append(json.loads(line))

    print(f"  Loaded {len(cases)} cases")

    # Format training examples
    def format_example(case):
        """
        Format case into training prompt/completion

        Input: Seed event + context
        Output: JSON array of outcomes with probabilities
        """
        # Build prompt
        prompt = f"""Given this historical event:

Event: {case['seed_event']}
Date: {case['seed_date']}
Context: {case['context']}

Predict the most likely outcomes over the next 12-24 months. For each outcome, provide:
- event: A specific, concrete event description
- probability: Likelihood (all probabilities should sum to 1.0)
- timeframe_months: Expected time until this event occurs

Output a JSON array of outcomes, ordered by probability (highest first).

Outcomes:"""

        # Build ground truth completion (what actually happened)
        # Format: simplified to just show the events that occurred
        outcomes = []
        for outcome in case['outcome_chain']:
            outcomes.append({
                "event": outcome['event'],
                "probability": 1.0 / len(case['outcome_chain']),  # Simplified: uniform
                "timeframe_months": outcome['timeframe_months']
            })

        completion = json.dumps(outcomes, indent=2)

        # Concatenate for causal LM training
        full_text = prompt + "\n" + completion + tokenizer.eos_token

        return {"text": full_text}

    formatted_data = [format_example(case) for case in cases]

    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)

    print(f"\n📝 Sample training example:")
    print(dataset[0]['text'][:500] + "...")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,  # Use bfloat16
        report_to="wandb" if wandb.run else "none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print(f"\n🏋️ Starting training for {num_epochs} epochs...")
    trainer.train()

    # Save final model
    print(f"\n💾 Saving model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")

    # Commit volume
    volume.commit()

    print(f"\n✅ SFT training complete!")
    print(f"  Model saved to: {output_dir}/final")

    return {
        "output_dir": f"{output_dir}/final",
        "num_cases": len(cases),
        "num_epochs": num_epochs,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
)
def prepare_data(local_data_path: str = "training/data/synthetic_cases.jsonl"):
    """
    Upload local training data to Modal volume

    Args:
        local_data_path: Path to local JSONL file
    """
    import shutil

    # Read local file
    with open(local_data_path) as f:
        data = f.read()

    # Write to volume
    volume_path = "/data/synthetic_cases.jsonl"
    with open(volume_path, 'w') as f:
        f.write(data)

    volume.commit()

    print(f"✅ Uploaded {local_data_path} to {volume_path}")

    return volume_path


def train_sft(*args, **kwargs):
    """Wrapper function for backward compatibility"""
    return train_sft_impl.remote(*args, **kwargs)


@app.local_entrypoint()
def main(
    upload_data: bool = False,  # Data already uploaded via test_volume
    run_training: bool = True,
    lora_rank: int = 64,
    num_epochs: int = 1,  # Default to 1 for quick testing
):
    """
    Main entry point for SFT training

    Usage:
        modal run training/modal_sft.py
        modal run training/modal_sft.py --lora-rank=32
        modal run training/modal_sft.py --num-epochs=3
    """
    print("🎯 PsychoHistory SFT Training Pipeline")
    print("=" * 60)

    # Step 1: Upload data (optional, already done via test_volume)
    if upload_data:
        print("\n📤 Uploading training data...")
        prepare_data.remote()

    # Step 2: Train
    if run_training:
        print(f"\n🏋️  Starting SFT training...")
        print(f"  LoRA Rank: {lora_rank}")
        print(f"  Epochs: {num_epochs}")

        result = train_sft_impl.remote(
            lora_rank=lora_rank,
            num_epochs=num_epochs
        )

        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"  Output: {result['output_dir']}")
        print(f"  Cases: {result['num_cases']}")
        print(f"  Epochs: {result['num_epochs']}")
        print(f"{'='*60}")
