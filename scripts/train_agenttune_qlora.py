import os
import sys
import json
import time
import math
import random
import torch
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTConfig, SFTTrainer

import agenttune
from agenttune.agentic import Project
from agenttune.agentic.events import EventLog, Event, EventKind
from agenttune.agentic.distill_trainer import DistillTrainer

sys.stdout.reconfigure(encoding='utf-8')

# Ensure reproducibility
random.seed(42)
torch.manual_seed(42)

BASE_MODEL_DIR = Path("artifacts/model/base").resolve()
OUTPUT_ADAPTER_DIR = Path("artifacts/model/repair_finetuned").resolve()
REPAIR_DATA_PATH = Path("data/repair/sanovocal_tool_contract_repair.jsonl").resolve()

def load_repair_eventlogs():
    """Load repair dataset and convert to AgentTune EventLog objects."""
    with open(REPAIR_DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    event_logs = []
    for r in records:
        log = EventLog(tier="full", id=r.get("id", ""))
        
        task_text = r.get("task", "")
        if not task_text and "messages" in r:
            for m in r["messages"]:
                if m.get("role") == "user":
                    task_text = m.get("content", "")
                    break
        log.append(Event(EventKind.OBSERVATION, {"text": task_text}))
        
        steps = r.get("steps", [])
        if steps:
            for i, step in enumerate(steps, 1):
                thought = step.get("thought")
                if thought:
                    log.append(Event(EventKind.REASONING, {"text": thought}))
                action = step.get("action", {})
                if action:
                    log.append(Event(EventKind.TOOL_CALL, {"action": action}))
                obs = step.get("observation")
                if obs:
                    log.append(Event(EventKind.TOOL_RESULT, {"output": obs}))
                log.append(Event(EventKind.TURN_COMPLETE, {"step": i}))
        elif "messages" in r:
            for m in r["messages"]:
                role = m.get("role")
                content = m.get("content", "")
                if role == "assistant":
                    if "<tool_call>" in content:
                        parts = content.split("<tool_call>")
                        thought = parts[0].strip()
                        if thought:
                            log.append(Event(EventKind.REASONING, {"text": thought}))
                        call_str = parts[1].split("</tool_call>")[0].strip()
                        try:
                            action = json.loads(call_str)
                        except Exception:
                            action = {"name": "unknown", "arguments": {}}
                        log.append(Event(EventKind.TOOL_CALL, {"action": action}))
                    else:
                        log.append(Event(EventKind.TEXT, {"text": content}))
                elif role == "tool":
                    log.append(Event(EventKind.TOOL_RESULT, {"output": content}))

        final_resp = r.get("final_response", "")
        if not final_resp and "messages" in r:
            for m in reversed(r["messages"]):
                if m.get("role") == "assistant" and "<tool_call>" not in m.get("content", ""):
                    final_resp = m.get("content", "")
                    break
        if final_resp:
            log.append(Event(EventKind.TEXT, {"text": final_resp}))
            
        log.append(Event(EventKind.REWARD, {"value": 1.0}, scope="episode"))
        event_logs.append(log)

    return event_logs

class SanoVocalQLoRATrainerFactory:
    """Custom trainer factory that loads the student model in 4-bit NF4,
    prepares it for k-bit training, applies PEFT exactly once,
    and returns a TRL SFTTrainer instance compatible with AgentTune."""

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] = None
    ):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def __call__(self, *, model, train_dataset, **kwargs):
        print("\n" + "=" * 70)
        print("SANOVOCAL QLORA TRAINER FACTORY INVOKED")
        print("=" * 70)

        # 2. Load Tokenizer
        print(f"Loading tokenizer from: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 1. Normalize dataset to HF Dataset with strict alternation repair for Cohere/Aya chat template
        rows = list(train_dataset)
        needs_repair = False
        for row in rows:
            try:
                self.tokenizer.apply_chat_template(row["messages"], tokenize=False)
            except Exception:
                needs_repair = True
                break

        if needs_repair:
            print("Applying AgentTune chat template strict-alternation fold for Aya...")
            from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user
            from agenttune.agentic.distill_trainer import _merge_consecutive_same_role

            rows = [
                {
                    **row,
                    "messages": _merge_consecutive_same_role(
                        fold_tool_messages_into_user(row["messages"])
                    ),
                }
                for row in rows
            ]

        train_dataset = Dataset.from_list(rows)
        print(f"[OK] Training dataset prepared with {len(train_dataset)} rows.")

        # 3. Configure 4-bit NF4 Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        print(f"Loading student model in 4-bit NF4: {model}")
        raw_model = AutoModelForCausalLM.from_pretrained(
            str(model),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Prepare for k-bit training (freezes base params, casts layer norms)
        raw_model = prepare_model_for_kbit_training(raw_model)

        # 4. LoRA Configuration
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules
        )

        print("Wrapping model with PEFT LoRA...")
        self.model = get_peft_model(raw_model, peft_config)

        # Dynamic trainable parameter count
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.3f}%)")

        assert trainable_params > 0, "No trainable parameters found!"

        # 5. Build SFTConfig
        sft_config_kwargs = dict(kwargs)
        if "max_seq_length" in sft_config_kwargs:
            sft_config_kwargs["max_length"] = sft_config_kwargs.pop("max_seq_length")
        sft_config_kwargs.setdefault("output_dir", str(OUTPUT_ADAPTER_DIR / "checkpoints"))
        sft_config_kwargs.setdefault("per_device_train_batch_size", 1)
        sft_config_kwargs.setdefault("gradient_accumulation_steps", 8)
        sft_config_kwargs.setdefault("learning_rate", 2e-4)
        sft_config_kwargs.setdefault("num_train_epochs", 3)
        sft_config_kwargs.setdefault("logging_steps", 5)
        sft_config_kwargs.setdefault("save_strategy", "no")
        sft_config_kwargs.setdefault("bf16", torch.cuda.is_available())
        sft_config_kwargs.setdefault("max_length", 512)
        sft_config_kwargs.setdefault("dataset_text_field", "text")

        sft_cfg = SFTConfig(**sft_config_kwargs)

        # 6. Instantiate TRL SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_cfg,
            train_dataset=train_dataset,
            processing_class=self.tokenizer
        )
        return self.trainer

def run_training():
    print("=" * 70)
    print("SANOVOCAL AGENTTUNE REPAIR FINE-TUNING PIPELINE")
    print("=" * 70)

    # 1. Load EventLogs
    logs = load_repair_eventlogs()
    print(f"Loaded {len(logs)} full-tier AgentTune EventLogs.")

    # 2. Build AgentTune Project
    project = Project()
    for log in logs:
        project.add_trajectory(log)
    print(f"Registered {len(logs)} trajectories in AgentTune Project.")

    # 3. Create DistillTrainer via official create_distill_trainer
    factory = SanoVocalQLoRATrainerFactory(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    trainer = agenttune.create_distill_trainer(
        student=str(BASE_MODEL_DIR),
        trajectories=logs,
        trainer_factory=factory,
        fmt="sft",
        output_dir=str(OUTPUT_ADAPTER_DIR),
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_available()
    )

    print("\nStarting genuine AgentTune Project.distill() training...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"\n[TRAINING FINISHED] Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # 4. Save Final Adapter
    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    factory.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    factory.tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    print(f"[SUCCESS] Final LoRA adapter saved to: {OUTPUT_ADAPTER_DIR}")

    return train_result

if __name__ == "__main__":
    run_training()
