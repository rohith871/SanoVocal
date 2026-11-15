"""
SanoVocal — Step 7: Dataset-to-Training Pipeline for CohereLabs/aya-expanse-8b
Converts AgentTune 300 train and 35 val trajectories into Aya-compliant
alternating chat conversations and verifies tokenizer encoding.
Does NOT touch the 45 held-out test trajectories.
"""

import sys
import json
from pathlib import Path
import huggingface_hub
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "CohereLabs/aya-expanse-8b"
SYSTEM_PROMPT = (
    "You are SanoVocal, a voice-first, safety-grounded Indian government health and welfare scheme navigator. "
    "Use tools when needed. Answer only from verified official sources. "
    "Do not diagnose, prescribe, or make final eligibility decisions. Escalate uncertain or urgent cases."
)

def trajectory_to_messages(traj: dict) -> list[dict]:
    """Formats an AgentTune trajectory into Aya's strictly alternating chat structure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": traj["task"]}
    ]
    for step in traj.get("steps", []):
        action = step.get("action")
        if action and action.get("name"):
            thought_text = step.get("thought", "").strip()
            action_json = json.dumps(action, ensure_ascii=False)
            assistant_content = f"<thought>\n{thought_text}\n</thought>\n<action>\n{action_json}\n</action>"
            messages.append({"role": "assistant", "content": assistant_content})
            
            obs_content = f"<observation>\n{step.get('observation', '')}\n</observation>"
            messages.append({"role": "user", "content": obs_content})
        else:
            obs = step.get("observation", "").strip()
            if obs:
                messages.append({"role": "assistant", "content": obs})
                # If there's another turn, must alternate
    
    # Final assistant answer
    final_resp = traj.get("final_response", "").strip()
    if messages[-1]["role"] == "assistant":
        # Merge if last message was assistant to preserve strict alternation
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{final_resp}"
    else:
        messages.append({"role": "assistant", "content": final_resp})
        
    return messages

def prepare_data():
    print("=" * 65)
    print(" SANOVOCAL — DATASET-TO-TRAINING PREPARATION FOR AYA EXPANSE 8B")
    print("=" * 65)

    token = huggingface_hub.get_token()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    traces_dir = ROOT / "data" / "traces"
    train_in = traces_dir / "agent_trajectories_train.jsonl"
    val_in = traces_dir / "agent_trajectories_val.jsonl"
    
    train_out = traces_dir / "aya_train_conversations.jsonl"
    val_out = traces_dir / "aya_val_conversations.jsonl"

    for in_path, out_path, expected_count in [(train_in, train_out, 300), (val_in, val_out, 35)]:
        print(f"\nProcessing {in_path.name} -> {out_path.name}...")
        records = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        
        assert len(records) == expected_count, f"Expected {expected_count} records, got {len(records)}"
        
        formatted_rows = []
        lengths = []
        for r in records:
            msgs = trajectory_to_messages(r)
            # Verify chat template applies with no error
            encoded = tokenizer.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
            if hasattr(encoded, "input_ids"):
                seq_len = encoded.input_ids.shape[1]
            elif isinstance(encoded, dict) and "input_ids" in encoded:
                seq_len = encoded["input_ids"].shape[1]
            else:
                seq_len = encoded.shape[1]
            lengths.append(seq_len)
            formatted_rows.append({
                "trajectory_id": r["trajectory_id"],
                "category": r["metadata"].get("category"),
                "language": r["metadata"].get("language"),
                "messages": msgs,
                "token_length": seq_len
            })
        
        with open(out_path, "w", encoding="utf-8") as f:
            for row in formatted_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        print(f"  Saved {len(formatted_rows)} formatted records.")
        print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f} tokens")
        print(f"  [OK] {out_path.name} validated against official Aya tokenizer.")

    print("\n" + "=" * 65)
    print(" TRAINING & VALIDATION DATASETS READY FOR AYA EXPANSE 8B QLoRA!")
    print(" (Held-out test set 45 remains untouched)")
    print("=" * 65)

if __name__ == "__main__":
    prepare_data()
