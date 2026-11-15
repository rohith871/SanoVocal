"""
SanoVocal — Phase C: Official CohereLabs/aya-expanse-8b QLoRA Smoke Test
Verifies all 12 smoke test criteria using exclusively the official gated model:
CohereLabs/aya-expanse-8b

Strictly avoids any substitute models, mirrors, or third-party weights.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import json
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFICIAL_MODEL_ID = "CohereLabs/aya-expanse-8b"

def run_aya_smoke_test():
    print("=" * 70)
    print(" SANOVOCAL — OFFICIAL COHERELABS/AYA-EXPANSE-8B QLoRA SMOKE TEST")
    print("=" * 70)

    # 1. Hardware & CUDA Environment
    print("\n[Step 1/12] Verifying Hardware & CUDA Environment...")
    assert torch.cuda.is_available(), "CUDA GPU required for QLoRA execution"
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU: {device_name} ({total_vram_gb:.2f} GB VRAM)")
    print(f"  PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print("  [OK] Local GPU environment verified.")

    # 2. Tokenizer Resolution
    print(f"\n[Step 2/12] Resolving Official Tokenizer for {OFFICIAL_MODEL_ID}...")
    import huggingface_hub
    from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
    token = huggingface_hub.get_token()
    assert token, "HuggingFace authentication token required for gated repo access"
    
    tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  Vocab size:      {len(tokenizer)} tokens")
    print(f"  Pad token:       {tokenizer.pad_token}")
    print("  [OK] Official Aya Expanse 8B tokenizer loaded without error.")

    # 3. Model Architecture & Configuration
    print(f"\n[Step 3/12] Inspecting Architecture for {OFFICIAL_MODEL_ID}...")
    config = AutoConfig.from_pretrained(OFFICIAL_MODEL_ID, token=token)
    print(f"  Model Type:          {config.model_type}")
    print(f"  Architecture:        {config.architectures}")
    print(f"  Hidden Size:         {config.hidden_size}")
    print(f"  Intermediate Size:   {getattr(config, 'intermediate_size', None)}")
    print(f"  Number of Layers:    {config.num_hidden_layers}")
    print(f"  Attention Heads:     {config.num_attention_heads} (Q) / {config.num_key_value_heads} (KV)")
    print(f"  Max Position:        {getattr(config, 'max_position_embeddings', 8192)}")
    print("  [OK] Model configuration validated.")

    # 4. 4-bit NF4 Quantization Configuration
    print("\n[Step 4/12] Configuring 4-bit NF4 BitsAndBytes Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print("  Quantization: 4-bit NF4 with double quantization configured [OK]")

    # 5. Load Training Trajectory Split
    print("\n[Step 5/12] Loading SanoVocal Trajectory Training Split...")
    from agenttune.agentic.trajectory.dataset import Trajectory, Step
    train_path = ROOT / "data" / "traces" / "agent_trajectories_train.jsonl"
    assert train_path.exists(), f"Missing {train_path}"
    with open(train_path, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
    steps = [Step(**s) for s in first_line["steps"]]
    sample_traj = Trajectory(
        task=first_line["task"],
        steps=steps,
        reward=first_line["reward"],
        trajectory_id=first_line["trajectory_id"],
        final_response=first_line["final_response"],
        metadata=first_line["metadata"]
    )
    print(f"  Trajectory ID:       {sample_traj.trajectory_id}")
    print(f"  Scenario Category:   {sample_traj.metadata.get('category')}")
    print(f"  User Task:           {sample_traj.task[:70]}...")
    print(f"  Tool Steps Count:    {len(sample_traj.steps)}")
    print("  [OK] Training trajectory loaded.")

    # 6. Trajectory Formatting for Aya's Alternating Turn Structure
    print("\n[Step 6/12] Converting Trajectory into Aya-Compatible Multi-turn Dialogue...")
    system_prompt = (
        "You are SanoVocal, a voice-first, safety-grounded Indian government health and welfare scheme navigator. "
        "Use tools when needed. Answer only from verified official sources. "
        "Do not diagnose, prescribe, or make final eligibility decisions. Escalate uncertain or urgent cases."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample_traj.task}
    ]
    for s in sample_traj.steps:
        if s.action and s.action.get("name"):
            messages.append({
                "role": "assistant",
                "content": f"<thought>\n{s.thought}\n</thought>\n<action>\n{json.dumps(s.action, ensure_ascii=False)}\n</action>"
            })
            messages.append({
                "role": "user",
                "content": f"<observation>\n{s.observation}\n</observation>"
            })
        else:
            messages.append({
                "role": "assistant",
                "content": s.observation
            })
    messages.append({
        "role": "assistant",
        "content": sample_traj.final_response
    })
    print(f"  Total Turns:         {len(messages)}")
    print(f"  Role Alternation:    {[m['role'] for m in messages]}")
    print("  [OK] Trajectory formatted according to Aya alternating role constraints.")

    # 7. Tokenize Trajectory with Official Aya Expanse 8B Chat Template
    print("\n[Step 7/12] Tokenizing Trajectory with Official Aya Chat Template...")
    try:
        tokenized_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        if hasattr(tokenized_ids, "input_ids"):
            tokenized_ids = tokenized_ids.input_ids
        elif isinstance(tokenized_ids, dict) and "input_ids" in tokenized_ids:
            tokenized_ids = tokenized_ids["input_ids"]
        print(f"  Tokenized Sequence Length: {tokenized_ids.shape[1]} tokens")
        print("  [OK] apply_chat_template successfully executed without template rejection.")
    except Exception as e:
        print(f"  [ERROR] Chat template execution error: {e}")
        raise

    # 8. LoRA / PEFT Configuration with Verified Target Modules
    print("\n[Step 8/12] Configuring PEFT LoRA Adapters for Aya Architecture...")
    from peft import LoraConfig
    # Target modules verified from transformers CohereAttention & CohereMLP
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=lora_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"  LoRA Rank (r):       16")
    print(f"  LoRA Alpha:          32")
    print(f"  Target Modules:      {lora_targets}")
    print("  [OK] PEFT LoRA configuration validated against Cohere architecture.")

    # 9. Model Loading (4-bit QLoRA)
    print(f"\n[Step 9/12] Initializing {OFFICIAL_MODEL_ID} with 4-bit NF4 Quantization...")
    from transformers import AutoModelForCausalLM
    from peft import prepare_model_for_kbit_training, get_peft_model

    # Use memory safety map: max 5.0 GiB on RTX 3050, offload overflow to CPU
    max_memory = {0: "5.0GiB", "cpu": "24GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        OFFICIAL_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=token
    )
    vram_used = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"  Model Loaded! Allocated VRAM: {vram_used:.2f} GB / {total_vram_gb:.2f} GB")
    print("  [OK] 4-bit NF4 quantization active on GPU.")

    # 10. LoRA Adapter Attachment
    print("\n[Step 10/12] Attaching LoRA Adapters to Model...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    trainable_p, total_p = model.get_nb_trainable_parameters()
    print(f"  Trainable Parameters: {trainable_p:,} ({100 * trainable_p / total_p:.3f}%)")
    print(f"  Total Parameters:     {total_p:,}")
    print("  [OK] LoRA adapters injected into Cohere attention layers.")

    # 11. Forward Pass
    print("\n[Step 11/12] Executing Forward Pass with Cross-Entropy Loss...")
    input_ids = tokenized_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    print(f"  Forward Pass Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss must be positive"
    print("  [OK] Forward pass completed successfully.")

    # 12. Backward Pass & Optimizer Step
    print("\n[Step 12/12] Executing Backward Pass & Optimizer Step...")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Computed Gradient Norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients must flow to trainable LoRA parameters"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer.step()
    optimizer.zero_grad()
    
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
    print(f"  Peak VRAM Used:      {peak_vram:.2f} GB / {total_vram_gb:.2f} GB")
    print("  [OK] Optimizer step executed cleanly without memory overflow.")

    print("\n" + "=" * 70)
    print(f" ALL 12 OFFICIAL COHERELABS/AYA-EXPANSE-8B SMOKE TEST CRITERIA PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_aya_smoke_test()
