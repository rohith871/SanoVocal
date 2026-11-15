import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/repair/sanovocal_tool_contract_repair.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total repair records: {len(records)}")
print("\nSample record 0 keys:")
print(list(records[0].keys()))

categories = {}
languages = {}
for r in records:
    c = r.get("category", "unknown")
    l = r.get("language", "unknown")
    categories[c] = categories.get(c, 0) + 1
    languages[l] = languages.get(l, 0) + 1

print("\nCategories breakdown:")
for c, cnt in sorted(categories.items()):
    print(f"  {c:35s}: {cnt}")

print("\nLanguages breakdown:")
for l, cnt in sorted(languages.items()):
    print(f"  {l:15s}: {cnt}")

print("\nInspect record 0 details:")
r0 = records[0]
for k in ["id", "category", "language", "scenario_type", "expected_tool"]:
    if k in r0:
        print(f"  {k}: {r0[k]}")

if "messages" in r0:
    print(f"\nMessages in record 0 ({len(r0['messages'])} turns):")
    for m in r0["messages"]:
        print(f"  Role: {m.get('role')}")
        print(f"  Content: {str(m.get('content'))[:150]}")
        if "tool_calls" in m:
            print(f"  Tool Calls: {m.get('tool_calls')}")
elif "conversations" in r0:
    print(f"\nConversations in record 0 ({len(r0['conversations'])} turns):")
    for c in r0["conversations"]:
        print(f"  From: {c.get('from')}")
        print(f"  Value: {str(c.get('value'))[:150]}")
elif "steps" in r0:
    print(f"\nSteps in record 0 ({len(r0['steps'])} steps):")
    print(json.dumps(r0["steps"][:2], indent=2))
