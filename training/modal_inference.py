"""
Quick inference test to see trained model outputs
"""

import modal
import json

app = modal.App("psychohistory-inference-test")

# Same volume and image as training
volume = modal.Volume.from_name("psychohistory-data", create_if_missing=True)

# Reuse the training image (has all dependencies)
from modal_sft import image

@app.function(
    image=image,
    gpu="H100",  # Fastest GPU for faster inference
    timeout=600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_inference():
    """Load trained adapter and generate predictions for a test case"""
    import os
    from unsloth import FastLanguageModel
    import torch

    # Handle HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    print("="*80)
    print("TESTING TRAINED MODEL INFERENCE")
    print("="*80)

    # Load base model
    print("\n1. Loading base model with trained adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters (same config as training)
    print("\n2. Adding LoRA adapters (rank 64)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load trained weights
    print("\n3. Loading trained weights from /data/models/sft/final...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "/data/models/sft/final")

    FastLanguageModel.for_inference(model)  # Enable inference mode

    print("✅ Model loaded with trained adapter!\n")

    # Test prompt - exactly like training format with emphasis on non-uniform distribution
    test_prompt = """Initial Event: U.S. Federal Reserve raises interest rates by 0.75%
Path so far: U.S. Federal Reserve raises interest rates by 0.75%
Current Event: U.S. Federal Reserve raises interest rates by 0.75%
Depth: 1/3
Timeframe: next 3 months

Research:
Historical analysis shows that aggressive rate hikes typically lead to immediate market reactions. The 2022 Fed rate increases of similar magnitude resulted in bond yield increases, mortgage rate spikes, and stock market volatility. Previous 75 basis point hikes were followed by housing market slowdowns within 2-3 months.

Predict 1-5 possible next events following from the current situation.

Requirements:
- Probabilities sum to 1.0
- Use NON-UNIFORM probabilities (e.g., 0.42, 0.28, 0.18, 0.12 - NOT 0.25, 0.25, 0.25)
- Assign higher probabilities to more likely outcomes based on research evidence
- DO NOT use uniform or overly rounded probabilities (avoid 0.2, 0.3, 0.5 patterns)
- Example good distribution: [0.38, 0.31, 0.19, 0.12] - varied and realistic
- Specific, measurable outcomes

Output JSON only:
[{"event": "...", "probability": 0.38}]

"""

    print("="*80)
    print("TEST INPUT:")
    print("="*80)
    print(test_prompt)

    # Generate
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS...")
    print("="*80)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # Let it generate fully, hoping for JSON at end
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract completion (after prompt)
    completion = generated_text[len(test_prompt):]

    print("\n" + "="*80)
    print("MODEL OUTPUT (RAW):")
    print("="*80)
    print(completion)

    # Try to parse JSON
    print("\n" + "="*80)
    print("PARSED PREDICTIONS:")
    print("="*80)

    try:
        # Extract JSON from output
        json_str = completion.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Find JSON array
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        if start != -1 and end > start:
            json_str = json_str[start:end]

        predictions = json.loads(json_str)

        print("\n✅ Successfully parsed predictions!\n")

        total_prob = 0
        for i, pred in enumerate(predictions):
            print(f"{i+1}. Event: {pred.get('event', 'N/A')}")
            print(f"   Probability: {pred.get('probability', 0):.3f}")
            total_prob += pred.get('probability', 0)
            print()

        print(f"Total probability: {total_prob:.3f}")

        if abs(total_prob - 1.0) < 0.1:
            print("✅ Probabilities sum to ~1.0 (good calibration!)")
        else:
            print("⚠️  Probabilities don't sum to 1.0")

        # Check if non-uniform
        probs = [p.get('probability', 0) for p in predictions]
        if len(set(probs)) > 1:
            print("✅ Non-uniform distribution (model learned to discriminate!)")
        else:
            print("⚠️  Uniform distribution")

        return predictions

    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        print("Raw completion:", completion[:500])
        return None


@app.local_entrypoint()
def main():
    """Run inference test"""
    predictions = test_inference.remote()

    print("\n" + "="*80)
    print("INFERENCE TEST COMPLETE")
    print("="*80)

    if predictions:
        print(f"\n✅ Model generated {len(predictions)} predictions")
        print(f"✅ Check output above for probability distribution")
    else:
        print("\n⚠️  Model output needs debugging")
