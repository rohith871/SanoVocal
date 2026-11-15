import json
import huggingface_hub
from transformers import AutoTokenizer, AutoConfig

OFFICIAL_MODEL_ID = "CohereLabs/aya-expanse-8b"
token = huggingface_hub.get_token()
tok = AutoTokenizer.from_pretrained(OFFICIAL_MODEL_ID, token=token)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

with open("data/traces/agent_trajectories_train.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

system_prompt = (
    "You are SanoVocal, a voice-first, safety-grounded Indian government health and welfare scheme navigator. "
    "Use tools when needed. Answer only from verified official sources. "
    "Do not diagnose, prescribe, or make final eligibility decisions. Escalate uncertain or urgent cases."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": sample["task"]}
]
for st in sample["steps"]:
    if st.get("action") and st["action"].get("name"):
        messages.append({
            "role": "assistant",
            "content": f"<thought>\n{st['thought']}\n</thought>\n<action>\n{json.dumps(st['action'], ensure_ascii=False)}\n</action>"
        })
        messages.append({
            "role": "user",
            "content": f"<observation>\n{st['observation']}\n</observation>"
        })
messages.append({"role": "assistant", "content": sample["final_response"]})

tokenized = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt")
print("Aya Chat Template successfully applied!")
print("Tokenized shape:", tokenized.input_ids.shape)
print("Decoded snippet:\n", tok.decode(tokenized.input_ids[0][:150]))
