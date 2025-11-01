# Working Training Configuration

**Status:** ✅ TRAINING SUCCESSFULLY!

**Date:** 2025-10-31
**App ID:** ap-90xBESKNsMV8wQ3j2veG1g

## Final Working Configuration

### Image Configuration

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Needed for git-based pip installs
    .pip_install(
        # Install torch 2.8+ first (has torch.int1 support for torchao)
        "torch>=2.8.0",
        "triton>=3.4.0",
        "torchvision",
        # Pin compatible versions for latest Unsloth
        "transformers>=4.51.3,<=4.57.2",  # Unsloth's supported range
        "trl==0.23.1",  # Latest Unsloth needs newer trl (but <=0.23.1)
        "peft>=0.7.1",
        "accelerate>=1.9.0",
        "datasets>=3.6.0",
        "bitsandbytes>=0.43.0",
        "xformers>=0.0.27",
        "wandb>=0.16.0",
    )
    .run_commands(
        # Install latest Unsloth from git (supports gpt-oss-20b, works with torch 2.8+)
        "pip install --no-cache-dir --upgrade 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
        "pip install --no-cache-dir --upgrade 'git+https://github.com/unslothai/unsloth-zoo.git'",
    )
)
```

### GPU Configuration

```python
@app.function(
    gpu="A10G",  # 24GB VRAM sufficient (Unsloth uses ~14GB)
    timeout=7200,
)
```

### Model Configuration

```python
model_name="unsloth/gpt-oss-20b",  # ✅ Use Unsloth's optimized variant!
lora_rank=8,  # Can go up to 64 for more capacity
```

## Training Results (1 Epoch Test Run)

```
Num examples = 10
Num Epochs = 1
Batch size = 1
Gradient accumulation = 16
Total batch size = 16

Trainable parameters = 3,981,312 of 20,918,738,496 (0.02% trained)

Training time: 137.6 seconds (~2.3 minutes)
Final loss: 3.8716

✅ Model saved to: /data/models/sft/final
```

## Memory Usage

**Actual VRAM:** ~14GB (as predicted!)
**vs vanilla:** 76GB+ (5.4x reduction)
**GPU:** A10G-24GB (plenty of headroom)

## Versions That Work Together

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.8.0 | Has torch.int1 for torchao |
| triton | 3.4.0 | Required by Unsloth |
| transformers | 4.57.1 | Within Unsloth's range |
| trl | 0.23.1 | Max version supported by latest Unsloth |
| peft | 0.17.1 | Latest compatible |
| unsloth | Latest (git) | Supports gpt-oss-20b |
| unsloth_zoo | Latest (git) | Matches unsloth version |

## Commands

### Run training
```bash
python3 -m modal run training/modal_sft.py --num-epochs=3 --lora-rank=64
```

### Monitor logs
```bash
python3 -m modal app list  # Get app ID
python3 -m modal app logs <APP_ID>
```

### Check saved models
```bash
python3 -m modal volume ls psychohistory-data
python3 -m modal run training/test_volume.py
```

## Key Insights

1. **Use unsloth/gpt-oss-20b** (not openai/gpt-oss-20b)
2. **Torch 2.8+ required** for torch.int1 support
3. **Latest Unsloth from git** for gpt-oss-20b support
4. **trl==0.23.1** (not 0.19.1 - too old for latest Unsloth)
5. **A10G is sufficient** (uses ~14GB, not 76GB+)

## Next Steps

1. ✅ 1-epoch test run completed
2. Run full training with `--num-epochs=3 --lora-rank=64`
3. Evaluate the model with `training/evaluation/evaluator.py`
4. Compare baseline vs SFT performance
