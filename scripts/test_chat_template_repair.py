import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from scripts.convert_repair_to_eventlogs import record_to_eventlog, records
from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user
from agenttune.agentic.distill_trainer import _merge_consecutive_same_role

tok = AutoTokenizer.from_pretrained('artifacts/model/base')
log = record_to_eventlog(records[0])
rows = log.as_dataset_rows('sft')
repaired_msgs = _merge_consecutive_same_role(fold_tool_messages_into_user(rows[0]['messages']))
print('Repaired messages count:', len(repaired_msgs))
for m in repaired_msgs:
    print(f"  [{m['role']}]: {m['content'][:80]}...")

rendered = tok.apply_chat_template(repaired_msgs, tokenize=False)
print('\n[SUCCESS] Rendered with chat template! Length:', len(rendered))
print(rendered[:300])
