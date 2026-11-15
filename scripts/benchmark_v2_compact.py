"""
SanoVocal V2 Compact Real Benchmark Script
Runs exactly 5 actual optimizer steps (20 micro-batches) using the genuine
AgentTune DistillTrainer pipeline on data/repair/sanovocal_tool_contract_repair_v2_compact.jsonl.
Measures: seconds/micro-batch, seconds/optimizer step, peak VRAM, GPU utilization, and loss.
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
base_model_dir = ROOT / "artifacts" / "model" / "base"
data_path = ROOT / "data" / "repair" / "sanovocal_tool_contract_repair_v2_compact.jsonl"
bench_output_dir = ROOT / "artifacts" / "model" / "_benchmark_compact"
bench_output_dir.mkdir(parents=True, exist_ok=True)

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

class BenchmarkTimerCallback(TrainerCallback):
    def __init__(self):
        self.step_times = []
        self.step_losses = []
        self.t_last = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.t_last = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.t_last is not None:
            dt = time.time() - self.t_last
            self.step_times.append(dt)
            # Record current loss from log history if available
            loss = None
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        loss = entry["loss"]
                        break
            self.step_losses.append(loss)
            vram_alloc = torch.cuda.memory_allocated() / (1024**2)
            vram_peak = torch.cuda.max_memory_allocated() / (1024**2)
            vram_res = torch.cuda.memory_reserved() / (1024**2)
            print(f"  [BENCHMARK STEP {state.global_step}] Time: {dt:.2f}s | Loss: {loss} | VRAM: {vram_alloc:.1f}MB (Peak: {vram_peak:.1f}MB, Res: {vram_res:.1f}MB)")

def run_benchmark():
    print("=" * 70)
    print("SANOVOCAL V2 COMPACT 5-STEP REAL BENCHMARK")
    print("=" * 70)

    # 1. Load compact EventLogs
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {data_path}")

    logs = []
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
        logs.append(log)

    project = Project()
    for log in logs:
        project.add_trajectory(log)

    # 2. Setup Trainer Factory
    class BenchmarkFactory:
        def __init__(self):
            self.model = None
            self.tokenizer = None
            self.trainer = None
            self.timer_cb = BenchmarkTimerCallback()

        def __call__(self, *, model, train_dataset, **kwargs):
            self.tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            raw_model = AutoModelForCausalLM.from_pretrained(
                str(model),
                quantization_config=bnb_config,
                device_map="auto"
            )
            raw_model = prepare_model_for_kbit_training(raw_model, use_gradient_checkpointing=True)

            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.model = get_peft_model(raw_model, peft_config)

            sft_config_kwargs = dict(kwargs)
            sft_config_kwargs["output_dir"] = str(bench_output_dir)
            sft_config_kwargs["per_device_train_batch_size"] = 1
            sft_config_kwargs["gradient_accumulation_steps"] = 4
            sft_config_kwargs["learning_rate"] = 2e-4
            sft_config_kwargs["max_steps"] = 5
            sft_config_kwargs["logging_steps"] = 1
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
                callbacks=[self.timer_cb]
            )
            return self.trainer

    factory = BenchmarkFactory()

    trainer = agenttune.create_distill_trainer(
        student=str(base_model_dir),
        trajectories=logs,
        trainer_factory=factory,
        output_dir=str(bench_output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=5,
        logging_steps=1,
        bf16=torch.cuda.is_available(),
        fp16=False
    )

    print("\nStarting 5 benchmark optimizer steps...")
    t_start = time.time()
    trainer.train()
    total_bench_time = time.time() - t_start

    step_times = factory.timer_cb.step_times
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    micro_batch_time = avg_step_time / 4

    total_1_epoch_steps = len(records) // 4  # 415 // 4 = 103 steps
    est_1_epoch_seconds = avg_step_time * total_1_epoch_steps
    est_1_epoch_minutes = est_1_epoch_seconds / 60
    est_1_epoch_hours = est_1_epoch_minutes / 60

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    res_vram = torch.cuda.max_memory_reserved() / (1024**2)

    gate_status = "GREEN" if est_1_epoch_hours <= 2.0 else ("YELLOW" if est_1_epoch_hours <= 2.5 else "RED")

    benchmark_report = {
        "benchmark_steps": len(step_times),
        "step_times_seconds": [round(t, 2) for t in step_times],
        "avg_step_time_seconds": round(avg_step_time, 2),
        "micro_batch_time_seconds": round(micro_batch_time, 2),
        "peak_vram_mb": round(peak_vram, 1),
        "reserved_vram_mb": round(res_vram, 1),
        "total_1_epoch_steps": total_1_epoch_steps,
        "est_1_epoch_minutes": round(est_1_epoch_minutes, 1),
        "est_1_epoch_hours": round(est_1_epoch_hours, 2),
        "gate_status": gate_status
    }

    report_path = ROOT / "artifacts" / "model" / "benchmark_compact_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)
    print(f"Average Optimizer Step Time: {avg_step_time:.2f} seconds ({micro_batch_time:.2f}s per micro-batch)")
    print(f"Peak VRAM:                   {peak_vram:.1f} MB / 6144 MB")
    print(f"Estimated 1-Epoch Steps:     {total_1_epoch_steps} optimizer steps")
    print(f"Estimated 1-Epoch Runtime:   {est_1_epoch_minutes:.1f} minutes ({est_1_epoch_hours:.2f} hours)")
    print(f"Runtime Gate Status:         {gate_status}")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
