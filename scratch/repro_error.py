import os
import sys
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

model_id = "ibm-granite/granite-vision-4.1-4b"
device = "cpu"

try:
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    print("Success!")
except Exception:
    import traceback
    traceback.print_exc()
