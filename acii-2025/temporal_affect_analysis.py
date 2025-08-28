import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lifelines import KaplanMeierFitter
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Path to the data file
DATA_PATH = 'acii-2025/grouped_by_user_id_sorted_by_timestamp_ensemble.json'

# EWMA smoothing parameter
ALPHA = 0.25

# Resampling interval (seconds)
DELTA_T = 5

# Laplace smoothing parameter for Markov transitions
BETA = 1.0

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

PLOT_DIR = 'acii-2025/temporal-plots'

def load_data(path=DATA_PATH):
    """Load the grouped JSON data into a dict."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def extract_sessions_by_student_session(data, session_gap_minutes=60):
    """
    Extract sessions grouped by (student_id, session_id) where sessions are defined
    as continuous interactions with gaps of at least session_gap_minutes.
    Returns a dict: {(student_id, session_id): [messages]}
    """
    sessions_by_group = defaultdict(list)
    
    for user_id, messages in data.items():
        # Filter to only user messages and sort by timestamp
        user_msgs = [msg for msg in messages if msg.get('role') == 'user']
        if len(user_msgs) < 2:
            continue
            
        sorted_msgs = sorted(user_msgs, key=lambda x: x['timestamp'])
        
        # Split into sessions based on time gaps
        sessions = []
        current_session = [sorted_msgs[0]]
        
        for i in range(1, len(sorted_msgs)):
            current_time = pd.to_datetime(sorted_msgs[i]['timestamp'])
            prev_time = pd.to_datetime(sorted_msgs[i-1]['timestamp'])
            time_diff = current_time - prev_time
            
            # If gap is >= session_gap_minutes, start a new session
            if time_diff >= pd.Timedelta(minutes=session_gap_minutes):
                if len(current_session) > 1:  # Only keep sessions with at least 2 messages
                    sessions.append(current_session)
                current_session = [sorted_msgs[i]]
            else:
                current_session.append(sorted_msgs[i])
        
        # Add the last session if it has at least 2 messages
        if len(current_session) > 1:
            sessions.append(current_session)
        
        # Create session groups with sequential session IDs
        for session_idx, session_msgs in enumerate(sessions):
            session_id = f"session_{session_idx}"
            group_key = (user_id, session_id)
            sessions_by_group[group_key] = session_msgs
    
    return dict(sessions_by_group)


def compute_global_tertiles(sessions_by_group):
    """
    Compute global tertiles across all valence values from all sessions.
    This ensures consistent state definitions across all sessions.
    """
    all_valence_values = []
    
    # Collect all valence values from all sessions
    for group_key, session in sessions_by_group.items():
        valence_values = [msg['ensemble_analysis']['valence'] for msg in session]
        # Filter out None values
        valid_valence = [v for v in valence_values if v is not None]
        all_valence_values.extend(valid_valence)
    
    if len(all_valence_values) == 0:
        return None, None
    
    # Compute global tertiles
    tertiles = np.percentile(all_valence_values, [33.33, 66.67])
    return tertiles[0], tertiles[1]


def bin_valence_to_states(valence_values, low_tertile, high_tertile):
    """
    Bin valence values into discrete states using global tertiles.
    Returns list of state labels.
    """
    if len(valence_values) == 0:
        return []
    
    if low_tertile is None or high_tertile is None:
        return []
    
    states = []
    for v in valence_values:
        if v is None:
            states.append(None)  # Keep None values as None
        elif v < low_tertile:
            states.append('negative')
        elif v < high_tertile:
            states.append('neutral')
        else:
            states.append('positive')
    
    return states


def extract_transitions_within_session(states):
    """
    Extract pairwise transitions from a sequence of states.
    Drops the first turn (as specified in the fix).
    Returns list of (from_state, to_state) tuples.
    """
    if len(states) < 2:
        return []
    
    # Drop the first turn and create transitions from remaining turns
    transitions = []
    # Start from index 1 to drop the first turn, then create transitions
    for i in range(1, len(states) - 1):  # Go up to len-1 to have valid pairs
        from_state = states[i]
        to_state = states[i + 1]
        # Only include transitions where both states are valid (not None)
        if from_state is not None and to_state is not None:
            transitions.append((from_state, to_state))
    
    return transitions


def build_markov_transition_matrix(sessions_by_group, beta=BETA):
    """
    Build Markov transition matrix using the minimal fix approach:
    1. Group by (student_id, session_id)
    2. Compute global tertiles across all valence values
    3. For each group, bin valence → states using global tertiles, form pairwise transitions
    4. Concatenate all within-session N_ij counts
    5. Apply Laplace smoothing: P_ij = (N_ij + β) / Σ_k(N_ik + β)
    6. Drop first turn of every session from transition construction
    """
    # State mapping
    state_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
    idx_to_state = {0: 'negative', 1: 'neutral', 2: 'positive'}
    n_states = 3
    
    # Compute global tertiles first
    low_tertile, high_tertile = compute_global_tertiles(sessions_by_group)
    print(f"Global tertiles: low={low_tertile:.3f}, high={high_tertile:.3f}")
    
    # Initialize transition counts (start with 0, not beta)
    transition_counts = np.zeros((n_states, n_states))
    
    # Process each session group
    for group_key, session in sessions_by_group.items():
        # Extract valence values
        valence_values = [msg['ensemble_analysis']['valence'] for msg in session]
        
        # Bin valence to states using global tertiles
        states = bin_valence_to_states(valence_values, low_tertile, high_tertile)
        
        # Extract transitions (drops first turn automatically)
        transitions = extract_transitions_within_session(states)
        
        # Count transitions
        for from_state, to_state in transitions:
            if from_state in state_to_idx and to_state in state_to_idx:
                i, j = state_to_idx[from_state], state_to_idx[to_state]
                transition_counts[i, j] += 1
    
    # Apply Laplace smoothing: P_ij = (N_ij + β) / Σ_k(N_ik + β)
    # Add beta to all counts
    smoothed_counts = transition_counts + beta
    
    # Calculate transition probabilities with Laplace smoothing
    row_sums = smoothed_counts.sum(axis=1, keepdims=True)
    transition_matrix = smoothed_counts / row_sums
    
    return transition_matrix, smoothed_counts, state_to_idx, idx_to_state


def calculate_dwell_times(transition_matrix):
    """
    Calculate expected dwell times for each state.
    Dwell time = 1 / (1 - P_ii) where P_ii is self-transition probability.
    """
    dwell_times = 1 / (1 - np.diag(transition_matrix))
    return dwell_times


def resample_and_smooth(session, key='valence', alpha=ALPHA, delta_t=DELTA_T):
    """Resample a scalar time series to uniform grid and apply EWMA smoothing."""
    # Extract timestamps and values
    times = [pd.to_datetime(msg['timestamp']) for msg in session]
    values = [msg['ensemble_analysis'][key] for msg in session]
    if not times:
        return [], []
    # Create uniform grid
    t0, tN = times[0], times[-1]
    grid = pd.date_range(start=t0, end=tN, freq=f'{delta_t}s')
    # Nearest neighbor interpolation
    ts = pd.Series(values, index=times)
    ts_interp = ts.reindex(grid, method='nearest')
    # EWMA smoothing
    smoothed = ts_interp.ewm(alpha=alpha, adjust=False).mean()
    return grid, smoothed.values


def detect_frustration_episodes(smoothed_valence, grid, emotion_labels=None, negative_labels=None):
    """
    Detect frustration episodes and return their durations (in seconds).
    An episode is a maximal run where (emotion_label in negative_labels) or (smoothed valence < 4).
    """
    episodes = []
    in_episode = False
    start_time = None
    for i in range(len(grid)):
        v = smoothed_valence[i]
        label = emotion_labels[i] if emotion_labels is not None else None
        is_neg = (label in negative_labels) if (label and negative_labels) else False
        is_neg = is_neg or (v < 4)
        if is_neg and not in_episode:
            in_episode = True
            start_time = grid[i]
        elif not is_neg and in_episode:
            in_episode = False
            end_time = grid[i]
            episodes.append((start_time, end_time))
    # Handle episode running to end
    if in_episode and start_time is not None:
        episodes.append((start_time, grid[-1]))
    durations = [(end - start).total_seconds() for start, end in episodes]
    return durations


def survival_analysis(durations):
    """Kaplan-Meier estimator and exponential fit for episode durations."""
    kmf = KaplanMeierFitter()
    T = np.array(durations)
    E = np.ones_like(T)  # All observed
    kmf.fit(T, event_observed=E)
    # Exponential fit
    from scipy.stats import expon
    lambda_hat = 1 / np.mean(T)
    t_half = np.log(2) / lambda_hat
    return kmf, t_half, lambda_hat


def plot_smoothed_trajectories(smoothed_series, grid, n_bootstrap=1000, save_dir=PLOT_DIR):
    ensure_dir(save_dir)
    arr = np.stack([s for s in smoothed_series if len(s) == len(grid)])
    mean = arr.mean(axis=0)
    boot_means = np.array([
        resample(arr, replace=True, n_samples=arr.shape[0]).mean(axis=0)
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(boot_means, 2.5, axis=0)
    upper = np.percentile(boot_means, 97.5, axis=0)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(mean))  # Use turn index for x-axis
    plt.plot(x, mean, label='Mean Smoothed Valence')
    plt.fill_between(x, lower, upper, color='b', alpha=0.2, label='95% CI')
    plt.xlabel('Turn Index')
    plt.ylabel('Valence')
    plt.title('Smoothed Valence Trajectories by Turn Index (Mean ± 95% CI)')
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(save_dir, 'smoothed_valence_trajectories.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved plot: {fname}")


def plot_survival_curve(kmf, t_half=None, save_dir=PLOT_DIR):
    ensure_dir(save_dir)
    plt.figure(figsize=(8, 5))
    kmf.plot_survival_function()
    plt.axhline(0.5, color='r', linestyle='--', label='Half-life')
    if t_half is not None:
        plt.axvline(t_half, color='g', linestyle='--', label=f'$t_{{1/2}}$ = {t_half:.1f}s')
    plt.xlabel('Episode Duration (s)')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Survival Curve for Frustration Episodes')
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(save_dir, 'survival_curve.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved plot: {fname}")


def plot_markov_matrix(P, save_dir=PLOT_DIR):
    ensure_dir(save_dir)
    plt.figure(figsize=(6, 5))
    
    # Create heatmap with larger annotation text
    ax = sns.heatmap(P, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=['neg', 'neu', 'pos'],
                     yticklabels=['neg', 'neu', 'pos'],
                     annot_kws={'size': 14})  # Larger annotation text
    
    # Make axis labels larger
    plt.xlabel('To State', fontsize=14)
    plt.ylabel('From State', fontsize=14)
    
    # Make title larger
    plt.title('Markov Transition Matrix (Valence Tertiles)', fontsize=16)
    
    # Make tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fname = os.path.join(save_dir, 'markov_transition_matrix.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {fname}")


def plot_dwell_times(dwell, save_dir=PLOT_DIR):
    ensure_dir(save_dir)
    plt.figure(figsize=(6, 4))
    states = ['negative', 'neutral', 'positive']
    plt.bar(states, dwell)
    plt.ylabel('Expected Dwell Time (steps)')
    plt.title('Expected Dwell Time in Each State')
    plt.tight_layout()
    fname = os.path.join(save_dir, 'dwell_times.png')
    plt.savefig(fname)
    plt.close()
    print(f"Saved plot: {fname}")


def get_unique_emotion_labels(data):
    """
    Extract all unique emotion_label values from claude_sonnet_analysis, gpt4_analysis, and gemini_analysis.
    Returns a set of unique labels.
    """
    labels = set()
    for user_msgs in data.values():
        for msg in user_msgs:
            for key in ['claude_sonnet_analysis', 'gpt4_analysis', 'gemini_analysis']:
                analysis = msg.get(key, [])
                for entry in analysis:
                    label = entry.get('emotion_label')
                    if label:
                        labels.add(label.lower())
    return labels


def classify_negative_emotions(labels):
    """
    Use a simple ML approach to classify emotion labels as negative or not.
    For demonstration, use a keyword-based approach. For a real ML model, use zero-shot classification or train a classifier.
    Returns a dict: {label: is_negative}
    """
    # Example negative keywords
    negative_keywords = [
        'frustration', 'confusion', 'anxiety', 'anger', 'sadness', 'disappointment',
        'boredom', 'fear', 'embarrassment', 'shame', 'disgust', 'guilt', 'stress',
        'overwhelm', 'hopelessness', 'uncertainty', 'irritation', 'resentment',
        'helplessness', 'worry', 'discouragement', 'nervous', 'upset', 'tired', 'fatigue'
    ]
    is_negative = {}
    for label in labels:
        is_negative[label] = any(neg in label for neg in negative_keywords)
    return is_negative


def print_emotion_label_classification(data):
    labels = get_unique_emotion_labels(data)
    is_negative = classify_negative_emotions(labels)
    print("Unique emotion labels and negative/positive classification:")
    for label in sorted(labels):
        print(f"{label:20s} : {'NEGATIVE' if is_negative[label] else 'POSITIVE/NEUTRAL'}")


def save_unique_emotion_labels(data, filename='./acii-2025/unique_emotion_labels.txt'):
    labels = get_unique_emotion_labels(data)
    with open(filename, 'w') as f:
        for label in sorted(labels):
            f.write(label + '\n')
    print(f"Saved {len(labels)} unique emotion labels to {filename}")


def load_negative_emotion_labels(filename='./acii-2025/negative_emotion_labels.txt'):
    try:
        with open(filename, 'r') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: {filename} not found. No negative emotion labels loaded.")
        return set()


def interpolate_emotion_labels(session, grid, key='ensemble_analysis'):
    """
    Interpolate emotion labels to the resampled grid using nearest neighbor.
    Returns a list of labels aligned to grid.
    """
    times = [pd.to_datetime(msg['timestamp']) for msg in session]
    labels = []
    for msg in session:
        label = None
        if key in msg and msg[key] and isinstance(msg[key], dict):
            label = msg[key].get('emotion_label')
        if label is not None:
            labels.append(label.lower())
        else:
            labels.append(None)
    ts = pd.Series(labels, index=times)
    ts_interp = ts.reindex(grid, method='nearest')
    return ts_interp.values.tolist()


def save_negative_emotion_labels_guess(data, filename='./acii-2025/negative_emotion_labels.txt'):
    labels = get_unique_emotion_labels(data)
    is_negative = classify_negative_emotions(labels)
    with open(filename, 'w') as f:
        for label in sorted(labels):
            if is_negative[label]:
                # Only write the negative label, no comments
                f.write(f"{label}\n")
    print(f"Saved negative emotion label guesses to {filename}")


def main():
    data = load_data()
    print_emotion_label_classification(data)
    save_unique_emotion_labels(data)
    save_negative_emotion_labels_guess(data)
    negative_labels = load_negative_emotion_labels()
    
    # Use the new session extraction method with 60-minute gap definition
    sessions_by_group = extract_sessions_by_student_session(data, session_gap_minutes=60)
    print(f"Found {len(sessions_by_group)} session groups (defined by >=60 min gaps)")
    
    # Build Markov transition matrix using the minimal fix approach
    transition_matrix, smoothed_counts, state_to_idx, idx_to_state = build_markov_transition_matrix(sessions_by_group, beta=BETA)
    
    # Calculate dwell times
    dwell_times = calculate_dwell_times(transition_matrix)
    
    print(f"Raw transition counts matrix:")
    print(smoothed_counts - BETA)  # Show raw counts
    print(f"\nSmoothed transition counts matrix (with Laplace smoothing β={BETA}):")
    print(smoothed_counts)
    print(f"\nTransition probability matrix:")
    print(transition_matrix)
    print(f"\nDwell times: {dwell_times}")
    
    # For backward compatibility, also run the old analysis
    # Create a simple session extraction for the old analysis
    sessions = []
    for user_id, messages in data.items():
        # Filter to only user messages
        user_msgs = [msg for msg in messages if msg.get('role') == 'user']
        # Sort by timestamp
        sorted_msgs = sorted(user_msgs, key=lambda x: x['timestamp'])
        sessions.append(sorted_msgs)
    
    all_durations = []
    all_smoothed = []
    min_len = None
    ref_grid = None
    for session in sessions:
        grid, smoothed_valence = resample_and_smooth(session, key='valence')
        if len(smoothed_valence) < 2:
            continue
        all_smoothed.append(smoothed_valence)
        # Interpolate emotion labels to grid
        emotion_labels = interpolate_emotion_labels(session, grid, key='ensemble_analysis')
        durations = detect_frustration_episodes(smoothed_valence, grid, emotion_labels, negative_labels)
        all_durations.extend(durations)
        if min_len is None or len(smoothed_valence) < min_len:
            min_len = len(smoothed_valence)
            ref_grid = grid
    
    if all_smoothed and min_len is not None and ref_grid is not None:
        all_smoothed = [s[:min_len] for s in all_smoothed if len(s) >= min_len]
        kmf, _, _ = survival_analysis(all_durations)
        
        # Generate plots
        plot_smoothed_trajectories(all_smoothed, ref_grid)
        plot_survival_curve(kmf, None)
        plot_markov_matrix(transition_matrix)
        plot_dwell_times(dwell_times)

if __name__ == '__main__':
    main() 