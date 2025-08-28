# ------------------------------------------------------------
# pytutor_affect_analysis.py
# ------------------------------------------------------------
from __future__ import annotations

import json
import itertools
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------- 1. LOAD & FLATTEN ------------------------------------------------
DATA_PATH = Path("acii-2025") / "pytutor_chat_messages_with_ensemble.json"

with DATA_PATH.open("r", encoding="utf-8") as f:
    raw: list[dict] = json.load(f)

records: list[dict] = []
MODELS = ("claude_sonnet", "gpt4", "gemini")

for turn in raw:
    rec: dict = {
        "user_id":    turn["user_id"],
        "role":       turn["role"],                         # 'user' or 'assistant'
        "timestamp":  pd.to_datetime(turn["timestamp"]),
        "content":    turn["content"],
        "ens_label":  turn["ensemble_analysis"]["emotion_label"],
        "ens_val":    turn["ensemble_analysis"]["valence"],
        "ens_aro":    turn["ensemble_analysis"]["arousal"],
        "ens_learn":  turn["ensemble_analysis"]["learning"],
    }
    for m in MODELS:
        rec[f"{m}_labels"] = [a.get("emotion_label", "<none>") for a in turn[f"{m}_analysis"]]
    records.append(rec)

df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
# Filter to only student messages (role == 'user' means student)
df = df[df["role"] == "user"].reset_index(drop=True)

# Ensure relevant columns are numeric and print diagnostics if needed
if df.empty:
    print("[ERROR] DataFrame is empty after filtering for students (role == 'user'). Check your data.")
else:
    for col in ["ens_val", "ens_aro", "ens_learn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_nan = df[col].isna().sum()
        print(f"Column {col}: {n_nan} NaN values out of {len(df)}")
    print(df[["ens_val", "ens_aro", "ens_learn"]].head())

# ---------- 2. OVERALL DISTRIBUTIONS ----------------------------------------
PLOTS_PATH = Path("acii-2025") / "plots"
PLOTS_PATH.mkdir(exist_ok=True)

# Set larger font sizes for better legibility
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["ens_val"], bins=np.arange(0.5, 9.5, 1), alpha=0.6, label="Valence")
ax.hist(df["ens_aro"], bins=np.arange(0.5, 9.5, 1), alpha=0.6, label="Arousal")
ax.hist(df["ens_learn"], bins=np.arange(0.5, 9.5, 1), alpha=0.6, label="Learning")
ax.set_xlabel("Score (1–9)", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.set_title("Overall Distributions (Ensemble, Students Only)", fontsize=16)
ax.legend(loc="upper left", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(PLOTS_PATH / "overall_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

top10 = Counter(df["ens_label"]).most_common(10)
labels, counts = zip(*top10)
plt.figure(figsize=(10, 6))
plt.barh(labels[::-1], counts[::-1])
plt.title("Top-10 Emotion Labels (Ensemble, Students Only)", fontsize=16)
plt.xlabel("# Turns", fontsize=14)
plt.ylabel("Emotion Labels", fontsize=14)
plt.tick_params(axis='x', which='major', labelsize=16)  # Larger font for x-axis numbers
plt.tick_params(axis='y', which='major', labelsize=16)  # Larger font for emotion labels
plt.tight_layout()
plt.savefig(PLOTS_PATH / "top10_emotion_labels.png", dpi=300, bbox_inches='tight')
plt.close()

total_turns = len(df)
print("Top 10 Emotion Labels (Students Only):")
top10_pct_sum = 0.0
for label, count in top10:
    pct = 100 * count / total_turns
    top10_pct_sum += pct
    print(f"{label:20s} | {count:4d} turns | {pct:5.2f}%")
print(f"Sum of Top 10 Percentages: {top10_pct_sum:5.2f}%")

# ---------- 3. STUDENT HISTOGRAMS -------------------------------------
for metric, title in [("ens_val", "Valence"), ("ens_aro", "Arousal"), ("ens_learn", "Learning")]:
    plt.figure(figsize=(8, 5))
    plt.hist(df[metric],
             bins=np.arange(0.5, 9.5, 1),
             alpha=0.7,
             color="tab:blue")
    plt.title(f"{title} Distribution (Students Only)", fontsize=16)
    plt.xlabel("Score (1–9)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f"{metric}_students_only.png", dpi=300, bbox_inches='tight')
    plt.close()

# ---------- 4. ADDITIONAL HIGH-LEVEL ANALYSES -------------------------------
# 4-a) Unique emotion vocab size per source
unique_by_model = {
    m: len(set(itertools.chain.from_iterable(df[f"{m}_labels"]))) for m in MODELS
}
unique_by_model["ensemble"] = df["ens_label"].nunique()
print("Unique emotion labels (students only):", unique_by_model)

# 4-b) Inter-LLM agreement on PRIMARY (top-ranked) label
def first_label(lst: list[str]) -> str:        # helper
    return lst[0] if lst else "<none>"

primary = pd.DataFrame({
    m: df[f"{m}_labels"].map(first_label) for m in MODELS
})
# pair-wise agreement
for a, b in itertools.combinations(MODELS, 2):
    agreement = (primary[a] == primary[b]).mean()
    print(f"Agreement {a} vs {b} (students only): {agreement:.3%}")
# majority vote vs ensemble
majority_vote = primary.mode(axis=1)[0]
maj_vs_ens = (majority_vote == df["ens_label"]).mean()
print(f"Majority-vote LLMs vs Ensemble label (students only): {maj_vs_ens:.3%}")

# 4-c) Correlation matrix (Valence, Arousal, Learning)
corr = df[["ens_val", "ens_aro", "ens_learn"]].corr(method="pearson")
print("\nPearson correlation matrix (students only)\n", corr)

# 4-d) Valence ↔ Learning Pearson r
# Drop rows with NaN or inf in either column
valid = df[["ens_val", "ens_learn"]].replace([np.inf, -np.inf], np.nan).dropna()
r, p = pearsonr(valid["ens_val"], valid["ens_learn"])
print(f"\nValence–Learning correlation (students only): r = {r:.3f}, p = {p:.3e}")

# 4-e) Emotion-set entropy (diversity) for students only
entropy = -sum(p*np.log2(p) for p in (df["ens_label"].value_counts(normalize=True)))
print("\nShannon entropy (ensemble labels, students only):", entropy)

# 4-f) Temporal trend of mean Valence (7-day rolling)
df.set_index("timestamp", inplace=True)
mean_val = df["ens_val"].rolling("7D").mean()
plt.figure(figsize=(10, 5))
mean_val.plot()
plt.ylabel("7-day Mean Valence", fontsize=14)
plt.xlabel("Date", fontsize=14)
plt.title("Temporal Drift of Valence (Ensemble, Students Only)", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "temporal_drift_valence.png", dpi=300, bbox_inches='tight')
plt.close()

# Median and IQR for Valence, Arousal, Learning (students only)
def median_iqr(series):
    clean = series.dropna()
    print(f"{series.name}: {clean.size} non-NaN values")
    if clean.empty:
        return float('nan'), float('nan'), float('nan')
    med = np.median(clean)
    q1 = np.percentile(clean, 25)
    q3 = np.percentile(clean, 75)
    return med, q1, q3

med_v, q1_v, q3_v = median_iqr(df["ens_val"])
med_a, q1_a, q3_a = median_iqr(df["ens_aro"])
med_l, q1_l, q3_l = median_iqr(df["ens_learn"])

print(f"Median valence = {med_v:.0f} (IQR {q1_v:.0f}–{q3_v:.0f})")
print(f"Median arousal = {med_a:.0f} (IQR {q1_a:.0f}–{q3_a:.0f})")
print(f"Median learning = {med_l:.0f} (IQR {q1_l:.0f}–{q3_l:.0f})")

# ---------- 5. ROLE-SPECIFIC HISTOGRAMS (Learner vs Tutor) ------------------
# Reconstruct DataFrame with both roles
all_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
for col in ["ens_val", "ens_aro", "ens_learn"]:
    all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

roles = [("user", "Learner"), ("assistant", "Tutor")]
metrics = [
    ("ens_val", "Valence"),
    ("ens_aro", "Arousal"),
    ("ens_learn", "Learning"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=False)
for i, (role_key, role_label) in enumerate(roles):
    role_df = all_df[all_df["role"] == role_key]
    for j, (metric, metric_label) in enumerate(metrics):
        ax = axes[i, j]
        ax.hist(role_df[metric], bins=np.arange(0.5, 9.5, 1), alpha=0.7, color="tab:blue")
        ax.set_title(f"{role_label}: {metric_label}", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if i == 1:
            ax.set_xlabel("Score (1–9)", fontsize=12)
        if j == 0:
            ax.set_ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "role_specific_histograms.png", dpi=300, bbox_inches='tight')
plt.close()
