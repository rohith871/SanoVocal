"""
SanoVocal AgentTune Repair Fine-Tuning Pipeline V2 Compact
Trains Repair-FT V2 Compact using genuine AgentTune create_distill_trainer API,
4-bit NF4 QLoRA, and the validated Compact Dataset V2 (415 records).
Saves adapter to: artifacts/model/repair_finetuned_v2_compact/
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
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import SFTConfig, SFTTrainer

import agenttune
from agenttune.agentic import Project
from agenttune.agentic.events import Event, EventKind, EventLog

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/KOYA ROHITH/SanoVocal")
BASE_MODEL_DIR = ROOT / "artifacts" / "model" / "base"
OUTPUT_ADAPTER_DIR = ROOT / "artifacts" / "model" / "repair_finetuned_v2_compact"
REPAIR_DATA_PATH = ROOT / "data" / "repair" / "sanovocal_tool_contract_repair_v2_compact.jsonl"

COMPACT_SYSTEM_PROMPT = """You are SanoVocal, an AI assistant for Indian health and welfare schemes.
Emit tool calls strictly as: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
You have access to EXACTLY SEVEN official tools:
1. emergency_screen(symptoms: list[str]): Screen acute chest pain, stroke, breathing failure, severe trauma.
2. match_schemes(age: int, category: str, income_status: str): Discover/match eligible schemes from user profile.
3. get_scheme_rules(scheme_id: str): Fetch official eligibility rules and benefit limits for a known scheme.
4. get_documents(scheme_id: str): Retrieve required application paperwork and proof for a known scheme.
5. find_facility(district: str, facility_type: str): Locate empanelled hospitals, CSCs, or help desks in a district.
6. create_case_summary(collected_fields: dict): Compile structured user intake details before handoff.
7. escalate_to_human(case_summary: dict, reason: str): Hand off care denial, corruption, or complex disputes.
CRITICAL TOOL RULES:
- Never call get_schemes; it does not exist. Only use match_schemes, get_scheme_rules, or get_documents.
- match_schemes = scheme discovery; get_scheme_rules = scheme rules; get_documents = required papers.
SAFETY RULES: Never diagnose. Never prescribe drugs or dosage. Never invent schemes, benefits, or URLs. For acute emergencies, screen immediately. If no tool is required, reply directly."""

class ProgressETA_Callback(TrainerCallback):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.step_times = []
        self.t_last = None
        self.t_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_start = time.time()
        self.t_last = time.time()
        print(f"\n[TRAIN START] Total optimizer steps: {self.total_steps}")

    def on_step_begin(self, args, state, control, **kwargs):
        self.t_last = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.t_last is not None:
            dt = time.time() - self.t_last
            self.step_times.append(dt)
            step = state.global_step
            if step > 0 and (step % 5 == 0 or step == self.total_steps or step <= 3):
                elapsed = time.time() - self.t_start
                avg_dt = sum(self.step_times[-10:]) / len(self.step_times[-10:])
                rem_steps = max(0, self.total_steps - step)
                rem_seconds = rem_steps * avg_dt
                vram_alloc = torch.cuda.memory_allocated() / (1024**2)
                vram_peak = torch.cuda.max_memory_allocated() / (1024**2)
                loss_str = "N/A"
                if state.log_history:
                    for entry in reversed(state.log_history):
                        if "loss" in entry:
                            loss_str = f"{entry['loss']:.4f}"
                            break
                print(f"[STEP {step:3d}/{self.total_steps:3d}] "
                      f"Elapsed: {elapsed/60:5.1f}m | ETA: {rem_seconds/60:5.1f}m | "
                      f"Step: {dt:5.1f}s (avg10: {avg_dt:5.1f}s) | Loss: {loss_str} | "
                      f"VRAM: {vram_alloc:5.0f}MB (Peak: {vram_peak:5.0f}MB)")

def load_repair_eventlogs():
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

    return event_logs, records


class SanoVocalTrainerFactoryCompact:
    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] = None,
        total_steps: int = 103
    ):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.total_steps = total_steps
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def __call__(self, *, model, train_dataset, **kwargs):
        print("\n" + "=" * 70)
        print("SANOVOCAL TRAINER FACTORY V2 COMPACT INVOKED")
        print("=" * 70)

        # 1. Tokenizer
        print(f"Loading tokenizer from: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 2. Format dataset messages
        from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user
        from agenttune.agentic.distill_trainer import _merge_consecutive_same_role

        raw_rows = list(train_dataset)
        formatted_rows = []
        for row in raw_rows:
            msgs = row["messages"]
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": COMPACT_SYSTEM_PROMPT}] + msgs
            repaired_msgs = _merge_consecutive_same_role(fold_tool_messages_into_user(msgs))
            formatted_rows.append({"messages": repaired_msgs})

        train_ds = Dataset.from_list(formatted_rows)
        print(f"[OK] Training dataset prepared with {len(train_ds)} rows.")

        # 3. 4-bit NF4 Quantization
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
        raw_model = prepare_model_for_kbit_training(raw_model, use_gradient_checkpointing=True)

        # 4. LoRA Setup
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules
        )
        self.model = get_peft_model(raw_model, peft_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.3f}%)")

        # 5. SFTConfig
        sft_config_kwargs = dict(kwargs)
        sft_config_kwargs["output_dir"] = str(OUTPUT_ADAPTER_DIR / "checkpoints")
        sft_config_kwargs["per_device_train_batch_size"] = 1
        sft_config_kwargs["gradient_accumulation_steps"] = 4
        sft_config_kwargs["learning_rate"] = 2e-4
        sft_config_kwargs["num_train_epochs"] = 1
        sft_config_kwargs["logging_steps"] = 5
        sft_config_kwargs["save_strategy"] = "no"
        sft_config_kwargs["bf16"] = torch.cuda.is_available()
        sft_config_kwargs["fp16"] = False
        sft_config_kwargs["max_length"] = 1280

        sft_cfg = SFTConfig(**sft_config_kwargs)

        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_cfg,
            train_dataset=train_ds,
            processing_class=self.tokenizer,
            callbacks=[ProgressETA_Callback(total_steps=self.total_steps)]
        )
        return self.trainer


def run_training():
    print("=" * 70)
    print("SANOVOCAL REPAIR FINE-TUNING V2 COMPACT")
    print("=" * 70)
    print(f"Base Model:     {BASE_MODEL_DIR}")
    print(f"Dataset Path:   {REPAIR_DATA_PATH}")
    print(f"Output Adapter: {OUTPUT_ADAPTER_DIR}")

    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load EventLogs
    logs, records = load_repair_eventlogs()
    total_records = len(records)
    total_steps = total_records // 4  # 415 // 4 = 103 optimizer steps
    print(f"Loaded {total_records} records -> {total_steps} optimizer steps.")

    # 2. Build AgentTune Project
    project = Project()
    for log in logs:
        project.add_trajectory(log)
    print(f"Registered {len(logs)} trajectories in AgentTune Project.")

    # 3. Create DistillTrainer
    factory = SanoVocalTrainerFactoryCompact(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        total_steps=total_steps
    )

    trainer = agenttune.create_distill_trainer(
        student=str(BASE_MODEL_DIR),
        trajectories=logs,
        trainer_factory=factory,
        output_dir=str(OUTPUT_ADAPTER_DIR / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        bf16=torch.cuda.is_available(),
        fp16=False
    )

    # 4. Train
    print("\nStarting 1-Epoch AgentTune Training...")
    t_start = time.time()
    train_result = trainer.train()
    train_duration = time.time() - t_start
    print(f"\n[OK] Training completed in {train_duration:.1f}s ({train_duration/60:.2f} mins, {train_duration/3600:.2f} hours).")

    # 5. Save Adapter
    print(f"Saving final adapter to: {OUTPUT_ADAPTER_DIR}")
    factory.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    factory.tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))

    # Manifest
    final_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    manifest = {
        "pipeline": "AgentTune 1.1.3 QLoRA Repair-FT V2 Compact",
        "base_model": str(BASE_MODEL_DIR),
        "dataset_path": str(REPAIR_DATA_PATH),
        "dataset_records": total_records,
        "adapter_output_dir": str(OUTPUT_ADAPTER_DIR),
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "total_optimizer_steps": total_steps,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 1280,
        "train_duration_seconds": round(train_duration, 2),
        "train_duration_minutes": round(train_duration / 60, 2),
        "train_duration_hours": round(train_duration / 3600, 2),
        "final_loss": round(final_loss, 4) if final_loss is not None else None
    }

    manifest_path = OUTPUT_ADAPTER_DIR / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("REPAIR-FT V2 COMPACT TRAINING COMPLETE")
    print("=" * 70)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    run_training()
