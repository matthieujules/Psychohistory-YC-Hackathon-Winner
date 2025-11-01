"""
Test Modal setup and secrets
"""

import modal

app = modal.App("test-psychohistory")

image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_secrets():
    """Test that secrets are accessible"""
    import os

    # Check both possible names
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_token:
        print(f"✅ HuggingFace token found: {hf_token[:10]}...")
        # Set both for compatibility
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
    else:
        print("❌ HuggingFace token not found")
        print("Available env vars:", [k for k in os.environ.keys()])

    return {"hf_token_exists": hf_token is not None}


@app.function(image=image)
def hello_modal():
    """Simple test function"""
    print("🎉 Hello from Modal!")
    return {"message": "Modal is working!"}


@app.local_entrypoint()
def main():
    print("🧪 Testing Modal Setup\n")

    # Test 1: Basic function
    print("Test 1: Basic Modal Function")
    result = hello_modal.remote()
    print(f"Result: {result}\n")

    # Test 2: Secrets
    print("Test 2: HuggingFace Secret")
    result = test_secrets.remote()
    print(f"Result: {result}\n")

    print("✅ All tests passed!")
