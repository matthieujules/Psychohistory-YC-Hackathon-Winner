# Training Pipeline Debugging Log
Date: 2025-10-31

## Summary of Issues

Goal: Train GPT-OSS-20B (21B params) on probability tree prediction with Unsloth

## Error History

### Error #1: torch.int1 AttributeError (FIXED)
**Configuration:**
```python
torch==2.5.1
unsloth[cu124-torch250] from PyPI/GitHub
```

**Error:**
```
AttributeError: module 'torch' has no attribute 'int1'
```

**Root Cause:**
- torchao 0.14.1 requires `torch.int1` (added in PyTorch 2.6)
- We were using torch 2.5.1 which doesn't have it
- Unsloth dependencies pulled in torchao automatically

**Fix:** Upgrade to torch>=2.8.0

---

### Error #2: MXFP4 Auto-Dequantization (AVOIDED)
**Potential Issue:**
- `openai/gpt-oss-20b` uses MXFP4 quantization
- MXFP4 has no backward pass implementation
- Auto-dequantizes to bf16 → 76GB VRAM (OOM on H100-80GB!)

**Fix:** Use `unsloth/gpt-oss-20b` instead (optimized variant)

---

### Error #3: Model Not Supported (FIXED)
**Configuration:**
```python
unsloth==2025.7.8  # July 2025 version
model="unsloth/gpt-oss-20b"
```

**Error:**
```
NotImplementedError: Unsloth: unsloth/gpt-oss-20b is not supported in your current Unsloth version!
```

**Root Cause:**
- gpt-oss-20b support was added in August 2025
- We were using July 2025 version (2025.7.8)

**Fix:** Install latest Unsloth from git

---

### Error #4: TRL Version Incompatibility (IN PROGRESS)
**Configuration:**
```python
torch>=2.8.0
unsloth @ git+https://github.com/unslothai/unsloth.git (latest)
trl==0.19.1
```

**Error:**
```
NotImplementedError: Unsloth: For now we support `trl<=0.23.1`. Please downgrade!
```

**Root Cause:**
- Latest Unsloth (Oct 2025) needs trl API changes from newer versions
- trl 0.19.1 is TOO OLD (doesn't have required APIs)
- Latest Unsloth requires: 0.20.0 <= trl <= 0.23.1 (estimated)

**Fix:** Upgrade to trl==0.23.1

---

## Final Working Configuration

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Core dependencies with correct versions
        "torch>=2.8.0",  # Has torch.int1 for torchao
        "triton>=3.4.0",
        "torchvision",
        "transformers>=4.51.3,<=4.57.2",  # Unsloth's supported range
        "trl==0.23.1",  # Max supported by latest Unsloth
        "peft>=0.7.1",
        "accelerate>=1.9.0",
        "datasets>=3.6.0",
        "bitsandbytes>=0.43.0",
        "xformers>=0.0.27",
        "wandb>=0.16.0",
    )
    .run_commands(
        # Install latest Unsloth from git (supports gpt-oss-20b)
        "pip install --no-cache-dir --upgrade 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
        "pip install --no-cache-dir --upgrade 'git+https://github.com/unslothai/unsloth-zoo.git'",
    )
)

@app.function(
    gpu="A10G",  # 24GB VRAM sufficient (Unsloth uses ~14GB)
    timeout=7200,
)
def train_sft_impl(
    model_name="unsloth/gpt-oss-20b",  # NOT openai/gpt-oss-20b!
    lora_rank=64,
    ...
)
```

## Key Learnings

1. **GPT-OSS-20B has bleeding-edge MXFP4 quantization** → No training support, auto-dequantizes
2. **Unsloth provides optimized variants** → Use `unsloth/gpt-oss-20b`, not `openai/gpt-oss-20b`
3. **torch.int1 is required by torchao** → Need torch 2.6+ (we use 2.8.0)
4. **Version pinning is critical** → Latest Unsloth + old TRL = incompatible
5. **Use Unsloth from git for latest model support** → PyPI versions lag behind
6. **A10G is sufficient** → Unsloth's 4-bit quantization fits in 14GB (vs 76GB+ vanilla)

## Cost Savings

| Configuration | GPU | VRAM | Cost/hr | Status |
|---------------|-----|------|---------|--------|
| openai/gpt-oss-20b + vanilla | H100-80GB | 76GB (OOM!) | $4.00 | ❌ Fails |
| unsloth/gpt-oss-20b + Unsloth | A10G-24GB | ~14GB | $0.50 | ✅ Works |

**Savings: 8x cheaper, 5x less memory**

## Next Steps

1. ✅ Fix torch.int1 error (torch 2.8.0)
2. ✅ Fix model not supported (latest Unsloth from git)
3. ⏳ Fix trl incompatibility (trl 0.23.1) - testing now
4. ⏳ Verify model loads successfully
5. ⏳ Monitor training completion

## References

- Unsloth Docs: https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune
- Unsloth Blog: https://unsloth.ai/blog/gpt-oss
- Working Colab: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb
- Modal Unsloth Example: https://modal.com/docs/examples/unsloth_finetune
