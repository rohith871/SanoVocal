import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_DIR = Path("artifacts/model/base").resolve()

print("=" * 70)
print("SANOVOCAL QLORA FORWARD/BACKWARD SMOKE TEST ON RTX 3050")
print("=" * 70)

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

raw_model = AutoModelForCausalLM.from_pretrained(
    str(BASE_MODEL_DIR),
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 3. Prepare for k-bit training
raw_model = prepare_model_for_kbit_training(raw_model)

# 4. LoRA Config on attention projections
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(raw_model, peft_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_p = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {all_p:,} ({trainable/all_p*100:.3f}%)")
assert trainable > 0, "Zero trainable parameters!"

# 5. Synthetic batch forward + backward test
sample_text = "<|START_OF_TURN_TOKEN|><|user|>Patient has chest pain.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|assistant|><tool_call>{\"name\": \"emergency_screen\", \"arguments\": {\"symptoms\": [\"chest pain\"]}}</tool_call><|END_OF_TURN_TOKEN|>"
inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)
inputs["labels"] = inputs["input_ids"].clone()

print("\nRunning forward pass...")
outputs = model(**inputs)
loss = outputs.loss
print(f"Forward pass loss: {loss.item():.4f}")

print("Running backward pass...")
loss.backward()
print("Backward pass succeeded!")

# Verify gradient exists on LoRA parameter
lora_grad_found = False
for name, param in model.named_parameters():
    if "lora" in name and param.grad is not None:
        lora_grad_found = True
        break

assert lora_grad_found, "No gradients on LoRA parameters!"
print("Gradients successfully verified on LoRA parameters.")

# Check optimizer step
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
optimizer.step()
optimizer.zero_grad()
print("Optimizer step succeeded.")

vram_alloc = torch.cuda.memory_allocated(0) / (1024**3)
vram_res = torch.cuda.memory_reserved(0) / (1024**3)
print(f"\nPeak VRAM Allocated during smoke test: {vram_alloc:.2f} GiB / Reserved: {vram_res:.2f} GiB")
print("[PASS] QLoRA forward/backward smoke test 100% successful on RTX 3050!")
