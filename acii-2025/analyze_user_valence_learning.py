import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

USER_ID = "714605a907b047e0aa9998f5b4dfe268"
INPUT_PATH = Path("acii-2025/grouped_by_user_id_sorted_by_timestamp.json")

# Efficiently load only the relevant user's data
def load_user_messages(input_path, user_id):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if user_id not in data:
        raise ValueError(f"User ID {user_id} not found.")
    return data[user_id]

def segment_sessions(messages, max_gap_seconds=3600):
    """
    Assigns a session number to each message. A new session starts if the gap between messages > max_gap_seconds.
    Returns a list of (session_id, message) tuples.
    """
    sessions = []
    current_session = 0
    prev_time = None
    for msg in messages:
        ts = msg.get('timestamp')
        if not ts:
            sessions.append((current_session, msg))
            continue
        # Parse ISO 8601 timestamp
        try:
            t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            sessions.append((current_session, msg))
            continue
        if prev_time is not None and (t - prev_time).total_seconds() > max_gap_seconds:
            current_session += 1
        sessions.append((current_session, msg))
        prev_time = t
    return sessions

def extract_points(messages, analysis_field):
    points = []
    timestamps = []
    for msg in messages:
        if msg.get('role') != 'user':
            continue
        ts = msg.get('timestamp')
        analysis = msg.get(analysis_field, [])
        if analysis and isinstance(analysis, list):
            # Use the first emotion only for now
            emo = analysis[0]
            valence = emo.get('valence')
            learning = emo.get('learning')
            if valence is not None and learning is not None:
                points.append((valence, learning))
                timestamps.append(ts)
    return points, timestamps

def plot_valence_learning(points, timestamps, title, ax):
    if not points:
        ax.set_title(f"No data for {title}")
        return
    # Transform: center (5,5) to (0,0)
    x, y = zip(*[((vx-5), (vy-5)) for vx, vy in points])
    ax.scatter(x, y, c=range(len(points)), cmap='viridis', s=60, zorder=3)
    # Draw arrows between points
    for i in range(len(points) - 1):
        ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                    zorder=2)
    for i, (vx, vy) in enumerate(zip(x, y)):
        ax.text(vx, vy+0.2, str(i+1), fontsize=8, ha='center', va='bottom', zorder=4)
    # Draw axes
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    # Quadrant labels
    ax.text(2.5, 2.5, "I", fontsize=18, fontweight='bold', ha='center', va='center', alpha=0.3)
    ax.text(-2.5, 2.5, "II", fontsize=18, fontweight='bold', ha='center', va='center', alpha=0.3)
    ax.text(-2.5, -2.5, "III", fontsize=18, fontweight='bold', ha='center', va='center', alpha=0.3)
    ax.text(2.5, -2.5, "IV", fontsize=18, fontweight='bold', ha='center', va='center', alpha=0.3)
    # Axis labels
    ax.set_xlabel('Affect (Valence): Negative  \u2190   0   \u2192  Positive')
    ax.set_ylabel('Learning: Un-learning  \u2190   0   \u2192  Constructive Learning')
    ax.set_title(title)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.grid(True, linestyle='--', alpha=0.5)

def main():
    messages = load_user_messages(INPUT_PATH, USER_ID)
    sessions = segment_sessions(messages, max_gap_seconds=3600)
    # Group messages by session
    from collections import defaultdict
    session_dict = defaultdict(list)
    for session_id, msg in sessions:
        session_dict[session_id].append(msg)
    # For each session, plot both analyses
    for session_id in sorted(session_dict.keys()):
        session_msgs = session_dict[session_id]
        points_claude, ts_claude = extract_points(session_msgs, 'claude_sonnet_analysis')
        points_gpt4, ts_gpt4 = extract_points(session_msgs, 'gpt4_analysis')
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        plot_valence_learning(points_claude, ts_claude, f'Claude Sonnet Analysis (Session {session_id+1})', axs[0])
        plot_valence_learning(points_gpt4, ts_gpt4, f'GPT-4 Analysis (Session {session_id+1})', axs[1])
        plt.suptitle(f'User {USER_ID}: Valence vs Learning (role: user) - Session {session_id+1}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main() 