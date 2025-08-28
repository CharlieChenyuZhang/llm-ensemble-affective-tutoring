#!/usr/bin/env python3
# affect_semester_insights.py
# High‑level semester‑wide affect analyses for PyTutor study.

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────
TS_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"  # ISO format from the JSON sample


def load_messages(path: Path, model: str = "gpt4_analysis") -> pd.DataFrame:
    """Flatten JSON into a DataFrame keeping only the top affect tag for the specified model ('gpt4_analysis' or 'claude_sonnet_analysis')."""
    with path.open(encoding="utf-8") as fp:
        raw: List[Dict[str, Any]] = json.load(fp)

    rows: List[Dict[str, Any]] = []
    for rec in raw:
        if not rec.get(model):  # skip if missing or empty
            continue
        top = rec[model][0]
        # Check for required keys
        if not all(k in top for k in ("emotion_label", "valence", "arousal", "learning")):
            continue
        rows.append(
            {
                "user_id": rec["user_id"],
                "role": rec["role"],
                "ts": datetime.strptime(rec["timestamp"], TS_FMT),
                "emotion": top["emotion_label"].lower(),
                "val": int(top["valence"]),
                "ar": int(top["arousal"]),
                "learning": int(top["learning"]),
            }
        )
    if not rows:
        # Return empty DataFrame with correct columns if no data
        return pd.DataFrame(columns=["user_id", "role", "ts", "emotion", "val", "ar", "learning"])
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


# ────────────────────────────────────────────────────────────────────────────────
# 1. Dominant emotions by role across the semester
# ────────────────────────────────────────────────────────────────────────────────

def dominant_emotions(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("role")["emotion"].value_counts().rename("n").reset_index()
    dom = counts.loc[counts.groupby("role")["n"].idxmax()].reset_index(drop=True)
    return dom  # columns: role, emotion, n


# ────────────────────────────────────────────────────────────────────────────────
# 2. Within‑session spiral trajectories
# ────────────────────────────────────────────────────────────────────────────────

def quadrant(val: int, learning: int, v_mid: int = 5, l_mid: int = 5) -> int:
    """
    Map (valence, learning) → Picard quadrant:
    Quadrants are numbered:
        II | I
       ----+----
        III| IV
    Returns 2 for II, 1 for I, 3 for III, 4 for IV.
    """
    if val <= v_mid and learning > l_mid:
        return 2  # Quadrant II: Constructive Learning, Negative Affect
    elif val > v_mid and learning > l_mid:
        return 1  # Quadrant I: Constructive Learning, Positive Affect
    elif val <= v_mid and learning <= l_mid:
        return 3  # Quadrant III: Un-learning, Negative Affect
    else:
        return 4  # Quadrant IV: Un-learning, Positive Affect


def make_sessions(df: pd.DataFrame, gap_min: int = 30) -> pd.DataFrame:
    """Assign session id per (user_id, role) with a time‑gap heuristic."""
    df = df.copy()
    df["gap"] = (
        df.groupby(["user_id", "role"])["ts"].diff().fillna(pd.Timedelta(seconds=0))
    )
    df["new_sess"] = df["gap"] > pd.Timedelta(minutes=gap_min)
    df["sess_id"] = df.groupby(["user_id", "role"])["new_sess"].cumsum()
    return df.drop(columns=["gap", "new_sess"])


SPIRAL = [  # frustration → confusion → insight → satisfaction (quadrants 3→2→1→0)
    3,
    2,
    1,
    0,
]


def spiral_score(traj: List[int]) -> bool:
    """Return True if traj contains the full spiral subsequence in order."""
    idx = 0
    for q in traj:
        if q == SPIRAL[idx]:
            idx += 1
            if idx == len(SPIRAL):
                return True
    return False


def session_spiral_fit(df: pd.DataFrame) -> float:
    # Map quadrants
    df["quad"] = df.apply(lambda r: quadrant(r.val, r.learning), axis=1)
    sess = df.groupby(["user_id", "sess_id"])["quad"].apply(list)
    fits = sess.apply(spiral_score)
    return fits.mean()  # proportion of sessions fitting the spiral


# ────────────────────────────────────────────────────────────────────────────────
# 3. Half‑life of negative states
# ────────────────────────────────────────────────────────────────────────────────
NEG = {"frustration", "anger", "annoyance", "sadness", "confusion"}  # tweak!


def compute_half_life(df: pd.DataFrame, role: str = "user") -> float:
    """Median time (in minutes) until a negative episode ends."""
    # negative if val<=3 or explicit negative label
    df = df.copy()
    df = df[df["role"] == role]
    df["is_neg"] = (df.val <= 3) | df.emotion.isin(NEG)

    # Build episodes per user
    episodes: List[Tuple[datetime, datetime]] = []
    for uid, sub in df.groupby("user_id"):
        sub = sub.sort_values("ts")
        in_neg = False
        start: datetime | None = None
        for _, row in sub.iterrows():
            if row.is_neg and not in_neg:
                in_neg = True
                start = row.ts
            elif not row.is_neg and in_neg:
                episodes.append((start, row.ts))
                in_neg = False
        if in_neg:  # episode runs to last msg → right‑censor
            episodes.append((start, None))

    durations = []
    observed = []
    for s, e in episodes:
        if e is None:
            continue  # drop right‑censored for simple half‑life
        durations.append((e - s).total_seconds() / 60)  # minutes
        observed.append(1)

    if not durations:
        return float("nan")

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=observed)
    return float(kmf.median_survival_time_)


def plot_emotion_distribution(df: pd.DataFrame) -> None:
    """Plot emotion distribution for each role (user and assistant) and save to file."""
    roles = ["user", "assistant"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, role in enumerate(roles):
        sub = df[df["role"] == role]
        counts = sub["emotion"].value_counts().sort_values(ascending=False)
        axs[i].bar(counts.index, counts.values, color="skyblue")
        axs[i].set_title(f"Emotion Distribution: {role}")
        axs[i].set_xlabel("Emotion")
        axs[i].set_ylabel("Count")
        axs[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    out_path = Path(__file__).parent / "emotion_distribution.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Emotion distribution plot saved to {out_path}")
    plt.close(fig)


def plot_quadrant_distribution(df: pd.DataFrame, role: str = "user", suffix: str = "") -> None:
    """Plot and save the distribution of points in each Picard quadrant for the given role as a PNG."""
    labels = {1: "I (+Val, +Learn)", 2: "II (-Val, +Learn)", 3: "III (-Val, -Learn)", 4: "IV (+Val, -Learn)"}
    sub = df[df["role"] == role].copy()
    sub["quad"] = sub.apply(lambda r: quadrant(r.val, r.learning), axis=1)
    quad_counts = sub["quad"].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    bars = plt.bar([labels.get(q, str(q)) for q in quad_counts.index], quad_counts.values, color="mediumseagreen")
    plt.xlabel("Picard Quadrant")
    plt.ylabel("Count")
    plt.title(f"Distribution of Points in Picard Quadrants ({role}{' - ' + suffix if suffix else ''})")
    for bar, count in zip(bars, quad_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')
    out_path = Path(__file__).parent / f"quadrant_distribution_{role}{'_' + suffix if suffix else ''}.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Quadrant distribution plot saved to {out_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────────
# CLI glue
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "pytutor_chat_messages_with_llm_analysis.json"
    )
    if not json_path.exists():
        sys.exit(f"💥 File not found: {json_path}")

    # GPT-4 analysis
    df_gpt4 = load_messages(json_path, model="gpt4_analysis")
    df_gpt4 = make_sessions(df_gpt4)
    # Claude analysis
    df_claude = load_messages(json_path, model="claude_sonnet_analysis")
    df_claude = make_sessions(df_claude)

    # 1 Dominant emotions
    dom = dominant_emotions(df_gpt4)

    # 2 Spiral fit (students only)
    student_df = df_gpt4[df_gpt4.role == "user"].copy()
    spiral_prop = session_spiral_fit(student_df)

    # 3 Half‑life of negative states (students)
    hl_mins = compute_half_life(df_gpt4, role="user")

    # ── Prepare summary text
    summary_lines = []
    summary_lines.append("Dominant emotions across semester")
    summary_lines.append(dom.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Proportion of student sessions that follow spiral pattern:")
    summary_lines.append(f"{spiral_prop * 100:0.1f}%")
    summary_lines.append("")
    summary_lines.append("Half‑life of negative states (students):")
    if np.isnan(hl_mins):
        summary_lines.append("Not enough complete negative episodes to compute.")
    else:
        summary_lines.append(f"{hl_mins:0.1f} minutes")
    summary_lines.append("")

    # Emotion distribution summary
    for role in ["user", "assistant"]:
        sub = df_gpt4[df_gpt4["role"] == role]
        counts = sub["emotion"].value_counts().sort_values(ascending=False)
        unique_emotions = counts.index.tolist()
        summary_lines.append(f"Emotion distribution for {role}:")
        summary_lines.append(f"  Total unique emotions: {len(unique_emotions)}")
        for emotion, count in counts.items():
            summary_lines.append(f"    {emotion}: {count}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)

    # Print to console
    print("\n" + summary_text)

    # Save to file
    out_path = Path(__file__).parent / "affect_semester_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Summary text output saved to {out_path}")

    # Plot emotion distribution for user and assistant
    plot_emotion_distribution(df_gpt4)

    # Plot quadrant distribution for user role, for both models
    plot_quadrant_distribution(df_gpt4, role="user", suffix="gpt4")
    plot_quadrant_distribution(df_claude, role="user", suffix="claude")


if __name__ == "__main__":
    main()
