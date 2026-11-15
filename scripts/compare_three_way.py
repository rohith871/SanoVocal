import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

METRICS_DIR = Path("artifacts/metrics").resolve()
BASE_FILE = METRICS_DIR / "baseline_45.json"
FT_FILE = METRICS_DIR / "repair_ft_45.json"
ST_FILE = METRICS_DIR / "safetune_final_45.json"
OUT_REPORT = METRICS_DIR / "comparison_report.json"

def format_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"

def format_delta(v1, v2):
    if v1 is None or v2 is None:
        return "N/A"
    diff = (v2 - v1) * 100
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}%"

def generate_three_way_comparison():
    print("=" * 80)
    print("SANOVOCAL 3-WAY COMPARATIVE BENCHMARK (LOCKED 45-TEST SET)")
    print("=" * 80)

    with open(BASE_FILE, "r", encoding="utf-8") as f:
        base_data = json.load(f)["metrics"]
    with open(FT_FILE, "r", encoding="utf-8") as f:
        ft_data = json.load(f)["metrics"]
    with open(ST_FILE, "r", encoding="utf-8") as f:
        st_data = json.load(f)["metrics"]

    metric_keys = [
        ("Correct Tool Selection", "correct_tool_selection"),
        ("Tool-Call Validity", "tool_call_validity"),
        ("Forbidden Tool Calls Rate", "forbidden_tool_rate"),
        ("Emergency Safety Rate", "emergency_safety_rate"),
        ("Medical Non-Prescription", "medical_non_prescription_rate"),
        ("Medical Non-Diagnosis", "medical_non_diagnosis_rate"),
        ("Hallucination Resistance", "hallucination_resistance_rate"),
        ("Escalation Correctness", "escalation_correctness_rate"),
        ("Overall Safety Compliance", "overall_safety_compliance"),
    ]

    headers = ["Metric", "Base Aya", "Repair-FT", "+SafeTune", "FT Δ", "SafeTune Δ", "Final Δ"]
    rows = []

    print(f"\n| {headers[0]:<28} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<8} | {headers[5]:<10} | {headers[6]:<8} |")
    print(f"|:{'-'*28}-|-{'-'*10}:|-{'-'*10}:|-{'-'*10}:|-{'-'*8}:|-{'-'*10}:|-{'-'*8}:|")

    comparison_dict = {}

    for label, key in metric_keys:
        b_val = base_data.get(key)
        ft_val = ft_data.get(key)
        st_val = st_data.get(key)

        ft_delta = format_delta(b_val, ft_val)
        st_delta = format_delta(ft_val, st_val)
        final_delta = format_delta(b_val, st_val)

        row_str = f"| {label:<28} | {format_pct(b_val):>10} | {format_pct(ft_val):>10} | {format_pct(st_val):>10} | {ft_delta:>8} | {st_delta:>10} | {final_delta:>8} |"
        print(row_str)

        comparison_dict[label] = {
            "base": b_val,
            "repair_ft": ft_val,
            "safetune": st_val,
            "ft_delta": ft_delta,
            "safetune_delta": st_delta,
            "final_delta": final_delta
        }

    # Mean Latency
    b_lat = base_data.get("mean_latency_ms", 0)
    ft_lat = ft_data.get("mean_latency_ms", 0)
    st_lat = st_data.get("mean_latency_ms", 0)
    print(f"| {'Mean Latency (ms)':<28} | {b_lat:>9.1f}ms | {ft_lat:>9.1f}ms | {st_lat:>9.1f}ms | {ft_lat - b_lat:>+7.1f}ms | {st_lat - ft_lat:>+9.1f}ms | {st_lat - b_lat:>+7.1f}ms |")

    report_payload = {
        "dataset": "data/traces/agent_trajectories_test.jsonl",
        "total_test_scenarios": 45,
        "models_evaluated": {
            "base": base_data.get("model_label"),
            "repair_ft": ft_data.get("model_label"),
            "safetune": st_data.get("model_label")
        },
        "metrics_comparison": comparison_dict,
        "latency_ms": {
            "base": b_lat,
            "repair_ft": ft_lat,
            "safetune": st_lat
        }
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n[OK] Comparison report saved: {OUT_REPORT}")
    return report_payload

if __name__ == "__main__":
    generate_three_way_comparison()
