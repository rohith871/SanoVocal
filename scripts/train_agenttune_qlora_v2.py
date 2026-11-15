"""
SanoVocal AgentTune Repair Fine-Tuning Pipeline V2
Trains Repair-FT V2 using the genuine AgentTune create_distill_trainer API,
4-bit NF4 QLoRA, and the validated Dataset V2 (415 records with system prompt).
Saves adapter to: artifacts/model/repair_finetuned_v2/
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTConfig, SFTTrainer

import agenttune
from agenttune.agentic import Project
from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.distill_trainer import DistillTrainer

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_MODEL_DIR = ROOT / "artifacts" / "model" / "base"
OUTPUT_ADAPTER_DIR = ROOT / "artifacts" / "model" / "repair_finetuned_v2"
REPAIR_DATA_PATH = ROOT / "data" / "repair" / "sanovocal_tool_contract_repair_v2.jsonl"

from scripts.build_repair_dataset_v2 import SANOVOCAL_SYSTEM_PROMPT

def load_repair_eventlogs():
    """Load repair dataset V2 and convert to AgentTune EventLog objects."""
    with open(REPAIR_DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    event_logs = []
    for r in records:
        log = EventLog(tier="full", id=r.get("id", ""))
        task_text = r.get("task", "")
        log.append(Event(EventKind.OBSERVATION, {"text": task_text}))

        steps = r.get("steps", [])
        if steps:
            for i, step in enumerate(steps, 1):
                thought = step.get("thought", "")
                if thought:
                    log.append(Event(EventKind.REASONING, {"text": thought}))
                action = step.get("action", {})
                if action:
                    log.append(Event(EventKind.TOOL_CALL, {"action": action}))
                obs = step.get("observation", "{}")
                if obs:
                    log.append(Event(EventKind.TOOL_RESULT, {"output": obs}))
                log.append(Event(EventKind.TURN_COMPLETE, {"step": i}))

        final_resp = r.get("final_response", "")
        if final_resp:
            log.append(Event(EventKind.TEXT, {"text": final_resp}))

        log.append(Event(EventKind.REWARD, {"value": 1.0}, scope="episode"))
        event_logs.append(log)

    return event_logs


class SanoVocalQLoRATrainerFactoryV2:
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
        print("SANOVOCAL QLORA TRAINER FACTORY V2 INVOKED")
        print("=" * 70)

        # 1. Load Tokenizer
        print(f"Loading tokenizer from: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 2. Normalize dataset rows: inject SanoVocal system prompt into position 0 and fold tool calls
        from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user
        from agenttune.agentic.distill_trainer import _merge_consecutive_same_role

        raw_rows = list(train_dataset)
        formatted_rows = []
        for row in raw_rows:
            msgs = row["messages"]
            # Ensure system prompt is present
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": SANOVOCAL_SYSTEM_PROMPT}] + msgs

            repaired_msgs = _merge_consecutive_same_role(fold_tool_messages_into_user(msgs))
            formatted_rows.append({"messages": repaired_msgs})

        train_dataset = Dataset.from_list(formatted_rows)
        print(f"[OK] Training dataset prepared with {len(train_dataset)} rows.")
        print(f"Sample row 0 messages count: {len(train_dataset[0]['messages'])}")
        print(f"Sample row 0 roles: {[m['role'] for m in train_dataset[0]['messages']]}")

        # 3. Configure 4-bit NF4 Quantization (matching V1 exact settings)
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
            device_map="auto"
        )

        # 4. Prepare for k-bit training and apply LoRA
        print("Preparing model for k-bit training...")
        raw_model = prepare_model_for_kbit_training(raw_model)

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

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.3f}%)")

        # 5. Build SFTConfig with max_length=2048 to fit system prompt
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
        sft_config_kwargs["bf16"] = torch.cuda.is_available()
        sft_config_kwargs["fp16"] = False
        sft_config_kwargs["max_length"] = 2048

        sft_cfg = SFTConfig(**sft_config_kwargs)

        # 6. Instantiate TRL SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_cfg,
            train_dataset=train_dataset,
            processing_class=self.tokenizer
        )
        return self.trainer


def run_training_v2():
    print("=" * 70)
    print("SANOVOCAL AGENTTUNE REPAIR FINE-TUNING PIPELINE V2")
    print("=" * 70)
    print(f"Base Model:       {BASE_MODEL_DIR}")
    print(f"Dataset V2:       {REPAIR_DATA_PATH}")
    print(f"Output Adapter:   {OUTPUT_ADAPTER_DIR}")

    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load EventLogs
    logs = load_repair_eventlogs()
    print(f"Loaded {len(logs)} full-tier AgentTune EventLogs from V2.")

    # 2. Build AgentTune Project
    project = Project()
    for log in logs:
        project.add_trajectory(log)
    print(f"Registered {len(logs)} trajectories in AgentTune Project.")

    # 3. Create DistillTrainer via official agenttune.create_distill_trainer
    factory = SanoVocalQLoRATrainerFactoryV2(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    trainer = agenttune.create_distill_trainer(
        student=str(BASE_MODEL_DIR),
        trajectories=logs,
        trainer_factory=factory,
        output_dir=str(OUTPUT_ADAPTER_DIR / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        bf16=torch.cuda.is_available(),
        fp16=False
    )

    # 4. Train via DistillTrainer
    print("\nStarting genuine AgentTune DistillTrainer training...")
    t_start = time.time()
    train_result = trainer.train()
    train_duration = time.time() - t_start
    print(f"\n[OK] Training completed in {train_duration:.1f}s ({train_duration/60:.2f} mins).")

    # 5. Save the final V2 adapter
    print(f"Saving final Repair-FT V2 adapter to: {OUTPUT_ADAPTER_DIR}")
    factory.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    factory.tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))

    # Save training metrics manifest
    train_manifest = {
        "pipeline": "AgentTune 1.1.3 QLoRA Repair-FT V2",
        "base_model": str(BASE_MODEL_DIR),
        "dataset_path": str(REPAIR_DATA_PATH),
        "dataset_records": len(logs),
        "adapter_output_dir": str(OUTPUT_ADAPTER_DIR),
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 2048,
        "train_duration_seconds": round(train_duration, 2),
        "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None
    }
    with open(OUTPUT_ADAPTER_DIR / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)

    print(f"[SUCCESS] Repair-FT V2 model trained and saved successfully!")
    print(json.dumps(train_manifest, indent=2))

if __name__ == "__main__":
    run_training_v2()
