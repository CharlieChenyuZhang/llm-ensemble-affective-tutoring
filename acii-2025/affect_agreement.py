#!/usr/bin/env python3
# affect_agreement.py
# Compute Claude-vs-GPT-4 affect consistency across taxonomies.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import pearsonr, spearmanr


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def load_llm_affect(path: Path) -> pd.DataFrame:
    """
    Flatten the JSON so each row is one message with:
        - top Claude Sonnet label
        - top GPT-4 label
        - numeric valence / arousal
    Extend if you need Top-k overlap later.
    """
    with path.open(encoding="utf-8") as fp:
        raw: List[Dict[str, Any]] = json.load(fp)

    rows: List[Dict[str, Any]] = []
    for rec in raw:
        # Skip records with missing analysis
        if not rec.get("claude_sonnet_analysis") or not rec.get("gpt4_analysis"):
            print(f"[WARN] Skipping record with missing analysis for user_id={rec.get('user_id')} at ts={rec.get('timestamp')}")
            continue
        top_claude = rec["claude_sonnet_analysis"][0]
        top_gpt = rec["gpt4_analysis"][0]
        # Check for required keys in both analyses
        required_keys = ["emotion_label", "valence", "arousal"]
        if not all(k in top_claude for k in required_keys) or not all(k in top_gpt for k in required_keys):
            print(f"[WARN] Skipping record with missing keys in analysis for user_id={rec.get('user_id')} at ts={rec.get('timestamp')}")
            continue
        rows.append(
            {
                "user_id": rec["user_id"],
                "ts": rec["timestamp"],
                "claude_em": top_claude["emotion_label"].lower(),
                "gpt_em": top_gpt["emotion_label"].lower(),
                "claude_val": int(top_claude["valence"]),
                "gpt_val": int(top_gpt["valence"]),
                "claude_ar": int(top_claude["arousal"]),
                "gpt_ar": int(top_gpt["arousal"]),
            }
        )
    return pd.DataFrame(rows)


def quadrant(val: int, ar: int, v_mid: int = 5, a_mid: int = 5) -> int:
    """Map (valence, arousal) → quadrant 0-3 (clockwise, starting at ++)."""
    return (val > v_mid) * 2 + (ar > a_mid)


def sentiment_bucket(val: int) -> str:
    """Three-way sentiment from valence."""
    if val <= 3:
        return "neg"
    if val >= 7:
        return "pos"
    return "neu"


# ──────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    # --------------------------------------------------
    # CLI
    # --------------------------------------------------
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "pytutor_chat_messages_with_llm_analysis.json"
    )
    if not json_path.exists():
        sys.exit(f"💥 File not found: {json_path}")

    # --------------------------------------------------
    # Load & prep
    # --------------------------------------------------
    df = load_llm_affect(json_path)
    print(f"Loaded {len(df):,} messages from {json_path}\n")

    # ── 1 Discrete-emotion agreement
    pct_match = (df["claude_em"] == df["gpt_em"]).mean() * 100
    kappa_em = cohen_kappa_score(df["claude_em"], df["gpt_em"])

    # ── 2 Continuous valence / arousal correlations
    val_r, val_p = pearsonr(df["claude_val"], df["gpt_val"])
    ar_r, ar_p = pearsonr(df["claude_ar"], df["gpt_ar"])

    # ── 3 Kort–Picard quadrant overlap
    df["claude_quad"] = [
        quadrant(v, a) for v, a in zip(df["claude_val"], df["claude_ar"])
    ]
    df["gpt_quad"] = [quadrant(v, a) for v, a in zip(df["gpt_val"], df["gpt_ar"])]
    kappa_quad = cohen_kappa_score(df["claude_quad"], df["gpt_quad"])

    # ── 4 Sentiment buckets (derived from valence)
    df["claude_sent"] = df["claude_val"].apply(sentiment_bucket)
    df["gpt_sent"] = df["gpt_val"].apply(sentiment_bucket)
    kappa_sent = cohen_kappa_score(df["claude_sent"], df["gpt_sent"])

    # ───────────────────────────────────────────────────────────────
    # Summary table
    # ───────────────────────────────────────────────────────────────
    summary = pd.DataFrame(
        {
            "Metric": [
                "Exact emotion match (%)",
                "Cohen κ (emotion)",
                "Pearson r (valence)",
                "Pearson r (arousal)",
                "Cohen κ (quadrant)",
                "Cohen κ (sentiment)",
            ],
            "Value": [
                pct_match,
                kappa_em,
                val_r,
                ar_r,
                kappa_quad,
                kappa_sent,
            ],
        }
    )
    # prettier float formatting
    pd.options.display.float_format = "{:0.3f}".format
    print("Agreement summary\n──────────────────")
    print(summary.to_string(index=False))

    # Save summary to file
    summary_path = json_path.with_suffix('.affect_agreement_summary.tsv')
    summary.to_csv(summary_path, sep='\t', index=False)
    print(f"\nSummary saved to {summary_path}")

    # ── 5 (Opt.) confusion matrix for discrete emotions
    # Uncomment to inspect where models disagree.
    # cm = confusion_matrix(df["claude_em"], df["gpt_em"])
    # print("\nConfusion matrix (Claude rows x GPT cols):\n", cm)


if __name__ == "__main__":
    main()
