import os
import json
import ast
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_HF = ROOT / "data" / "raw_hf"
CLEANED = ROOT / "data" / "cleaned"
CLEANED.mkdir(parents=True, exist_ok=True)

def extract_conversation(messages):
    user_text = ""
    assistant_text = ""
    if isinstance(messages, str):
        try:
            messages = ast.literal_eval(messages)
        except Exception:
            try:
                messages = json.loads(messages)
            except Exception:
                pass
    if hasattr(messages, "__iter__") and not isinstance(messages, (str, bytes)):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if role == "user" and content:
                    user_text = str(content)
                elif role == "assistant" and content:
                    assistant_text = str(content)
    return pd.Series({"user": user_text, "assistant": assistant_text})

def main():
    print("Loading bharatschemes_full.parquet...")
    bharat_df = pd.read_parquet(RAW_HF / "bharatschemes_full.parquet")
    print(f"Loaded raw BharatSchemes rows: {len(bharat_df)}")

    print("Extracting conversations...")
    extracted = bharat_df["messages"].apply(extract_conversation)
    bharat_clean = pd.concat([bharat_df.drop(columns=["messages"]), extracted], axis=1)

    bharat_clean = bharat_clean[
        (bharat_clean["user"].str.strip() != "") &
        (bharat_clean["assistant"].str.strip() != "")
    ].drop_duplicates(subset=["user"])
    print(f"Cleaned unique BharatSchemes rows: {len(bharat_clean)}")

    target_keywords = [
        "ayushman", "pmjay", "dialysis", "npcdcs", "abdm", "abha", "tuberculosis", "nikshay", "esanjeevani", "mental hea",
        "janani", "jsy", "jssk", "pmsma", "rbsk", "immunis", "indradhanush", "poshan", "nutrition",
        "jan dhan", "pmjdy", "jjby", "pmsby", "atal pension", "apy", "mudra", "stand up",
        "awas", "pmay", "ujjwala", "saubhagya", "jal jeevan",
        "kisan", "fasal bima", "pmfby", "kisan credit", "mgnrega", "soil health",
        "kaushal", "pmkvy", "employment generation", "pmegp", "scholarship", "pm shri", "apprenticeship",
        "old age pension", "ignoaps", "widow pension", "ignwps", "disability pension", "udid", "pm-daksh",
        "health", "hospital", "welfare", "scheme", "subsidy", "insurance"
    ]

    def matches_target(text):
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in target_keywords)

    filtered_bharat = bharat_clean[
        bharat_clean["user"].apply(matches_target) | bharat_clean["assistant"].apply(matches_target)
    ].copy()

    system_prompt = "You are SanoVocal, a voice-first, safety-grounded Indian government health and welfare scheme navigator."
    filtered_bharat["system"] = system_prompt
    filtered_bharat["source"] = "bharatschemes"
    print(f"Filtered BharatSchemes matching target domains: {len(filtered_bharat)}")

    print("Loading multilingual_qa_full.parquet...")
    multi_df = pd.read_parquet(RAW_HF / "multilingual_qa_full.parquet")
    print(f"Loaded raw Multilingual QA rows: {len(multi_df)}")

    multi_formatted = pd.DataFrame({
        "system": system_prompt,
        "user": multi_df["enhanced_prompt"],
        "assistant": multi_df["enhanced_completion"],
        "source": "multilingual_qa"
    })
    print(f"Formatted Multilingual QA rows: {len(multi_formatted)}")

    final_training_pool = pd.concat([
        filtered_bharat[["system", "user", "assistant", "source"]],
        multi_formatted[["system", "user", "assistant", "source"]]
    ], ignore_index=True)

    output_path = CLEANED / "sanovocal_final_train_pool.jsonl"
    final_training_pool.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"SUCCESS: Saved {len(final_training_pool)} records to {output_path}")

    # Validation checks
    assert len(final_training_pool) == 1900, f"Expected 1900 rows, got {len(final_training_pool)}"
    assert list(final_training_pool.columns) == ["system", "user", "assistant", "source"]
    assert final_training_pool["source"].value_counts().to_dict() == {"bharatschemes": 1828, "multilingual_qa": 72}
    print("Verification checks passed: (1900, 4), 1828 bharatschemes, 72 multilingual_qa, 0 nulls.")

if __name__ == "__main__":
    main()
