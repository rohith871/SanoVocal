import hashlib
import os
import json

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

files_to_check = {
    'Repair dataset': ('data/repair/sanovocal_tool_contract_repair.jsonl', '0A0F6E6D588587FC39B5CA2828C18BC7FB684F0586A0CA5F35549E2037F74FCF'),
    'Locked 45 test set': ('data/traces/agent_trajectories_test.jsonl', '667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3'),
    'Train 300 traces': ('data/traces/agent_trajectories_train.jsonl', '760C95D3024B7FB6DBF7E2D24985070F6189B2AA7DE84FA1877F5144CCB0E031'),
    'Val 35 traces': ('data/traces/agent_trajectories_val.jsonl', '4D6F5194544FD73D796F9CA047C7E6DB19AAFBDECE5B5ECEF229858778AE644A')
}

print("=" * 70)
print("SANOVOCAL FROZEN DATASETS VERIFICATION")
print("=" * 70)
for name, (path, exp_hash) in files_to_check.items():
    if os.path.exists(path):
        actual_hash = sha256_file(path)
        status = "MATCH" if actual_hash == exp_hash else "MISMATCH"
        print(f"{name:20s}: [{status}] {path}")
        print(f"   Actual:   {actual_hash}")
        if status != "MATCH":
            print(f"   Expected: {exp_hash}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]
        print(f"   Records:  {len(lines)}")
    else:
        print(f"{name:20s}: [MISSING] {path}")
print("=" * 70)
