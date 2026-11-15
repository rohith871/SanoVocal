import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_DIR = Path("artifacts/model/base").resolve()

print("=" * 70)
print(f"TESTING OFFLINE LOADING OF {BASE_MODEL_DIR}")
print("=" * 70)

# 1. Load Tokenizer
t0 = time.time()
print("Loading tokenizer from local cache...")
tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[OK] Tokenizer loaded in {time.time() - t0:.2f}s. Vocab size: {len(tokenizer)}")

# 2. Configure 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. Load Model
t1 = time.time()
print("Loading model in 4-bit NF4 on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    str(BASE_MODEL_DIR),
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

vram_alloc = torch.cuda.memory_allocated(0) / (1024**3)
vram_res = torch.cuda.memory_reserved(0) / (1024**3)
print(f"[OK] Model loaded in {time.time() - t1:.2f}s!")
print(f"     VRAM Allocated: {vram_alloc:.2f} GiB")
print(f"     VRAM Reserved:  {vram_res:.2f} GiB")

# 4. Forward Generation Sanity Check
prompt = "<|START_OF_TURN_TOKEN|><|user|>What is the coverage limit of Ayushman Bharat PM-JAY?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\nPrompt: What is the coverage limit of Ayushman Bharat PM-JAY?")
t2 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
gen_text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print(f"Generation ({time.time() - t2:.2f}s):\n{gen_text}")
print("\n[SUCCESS] Offline loading and forward generation verified!")
