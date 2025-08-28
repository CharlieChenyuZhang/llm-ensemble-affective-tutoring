#!/usr/bin/env python3
"""
Ensemble-fuse valence, arousal, learning, and emotion labels for every
PyTutor turn that has LLM analyses.

The algorithm follows Section \ref{sec:ensemble} of the paper:

  • Stage 1 – Intra-model rank-weighted pooling  
  • Stage 2 – Inter-model averaging  
  • Stage 3 – Label aggregation with plurality + tie-breaks
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def rank_weights(k: int) -> List[float]:
    """
    Linearly–decaying weights w_r ∝ (K - r + 1), normalised s.t. Σw_r = 1.
    """
    denom: float = k * (k + 1) / 2  # Σ_{j=1..K}(K - j + 1)
    return [(k - r + 1) / denom for r in range(1, k + 1)]


def pool_within_model(items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """
    Collapse a single model's ranked list (length Kₘ ≤ 5) to weighted means
    (v̂ₘ, âₘ, l̂ₘ) using Eq. (1).
    Skips items missing 'valence', 'arousal', or 'learning'.
    """
    # Filter out items missing required keys
    filtered_items = [it for it in items if all(k in it for k in ("valence", "arousal", "learning"))]
    skipped = len(items) - len(filtered_items)
    if skipped > 0:
        print(f"[pool_within_model] Skipped {skipped} items missing required keys.")
    k = len(filtered_items)
    if k == 0:
        return None, None, None
    w = rank_weights(k)
    v = sum(w_r * it["valence"] for w_r, it in zip(w, filtered_items))
    a = sum(w_r * it["arousal"] for w_r, it in zip(w, filtered_items))
    l = sum(w_r * it["learning"] for w_r, it in zip(w, filtered_items))
    return v, a, l


def plurality_label(model_lists: Dict[str, List[Dict[str, Any]]],
                    bar_v: float) -> str:
    """
    Determine e* (Stage 3):
      – count how many *models* include each label anywhere in their list;
      – choose the plurality winner; on ties, pick the label whose model-mean
        valence is closer to bar_v (higher wins); if still tied, lexicographic.
    """
    # track labels present in each model (ignore rank here)
    presence: Counter[str] = Counter()
    label_valence: defaultdict[str, List[float]] = defaultdict(list)

    for items in model_lists.values():
        if not items:
            continue
        # Only include items with 'emotion_label'
        labels_in_model = {it["emotion_label"] for it in items if "emotion_label" in it}
        if not labels_in_model:
            continue
        for lbl in labels_in_model:
            presence[lbl] += 1
        # use *rank-weighted* valence as proxy for each label's valence in model
        v_model, *_ = pool_within_model(items)
        for lbl in labels_in_model:
            label_valence[lbl].append(v_model)

    # plurality count
    if not presence:
        return None
    top_count = max(presence.values())
    candidates = [lbl for lbl, c in presence.items() if c == top_count]

    if len(candidates) == 1:
        return candidates[0]

    # tie-break #1 – higher ensemble-mean valence
    val_mean: Dict[str, float] = {lbl: sum(vs) / len(vs)
                                  for lbl, vs in label_valence.items()}
    max_val = max(val_mean[lbl] for lbl in candidates)
    candidates = [lbl for lbl in candidates if val_mean[lbl] == max_val]

    # tie-break #2 – lexicographic
    return sorted(candidates)[0]


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
INPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_gemini_analysis.json")
OUTPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_ensemble.json")

MODEL_KEYS = {
    "claude_sonnet_analysis",
    "gpt4_analysis",
    "gemini_analysis",
}

def fuse_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a *new* dict with ensemble fields appended to the original record."""
    # Filter only models that appear in this turn
    model_lists = {k: rec.get(k, []) for k in MODEL_KEYS if rec.get(k)}

    # Stage 1 & 2: numeric fusion
    pooled = [pool_within_model(lst) for lst in model_lists.values()]
    pooled_valid = [p for p in pooled if p[0] is not None and p[1] is not None and p[2] is not None]
    if pooled_valid:
        bar_v = sum(p[0] for p in pooled_valid) / len(pooled_valid)
        bar_a = sum(p[1] for p in pooled_valid) / len(pooled_valid)
        bar_l = sum(p[2] for p in pooled_valid) / len(pooled_valid)
    else:  # fallback if no analysis is present
        bar_v = bar_a = bar_l = None

    # Stage 3: label fusion
    label_star = plurality_label(model_lists, bar_v) if model_lists else None

    # Attach new fields
    rec_out = rec.copy()
    rec_out["ensemble_analysis"] = {
        "valence": bar_v,
        "arousal": bar_a,
        "learning": bar_l,
        "emotion_label": label_star,
    }
    return rec_out


def main() -> None:
    turns: List[Dict[str, Any]] = json.loads(INPUT_PATH.read_text())

    fused_turns = [fuse_record(turn) for turn in turns]

    OUTPUT_PATH.write_text(json.dumps(fused_turns, indent=2))
    print(f"✅  Ensemble-augmented file written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
