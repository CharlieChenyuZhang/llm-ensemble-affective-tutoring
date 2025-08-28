import json
from datetime import datetime, timedelta
from collections import Counter
import numpy as np

# Path to your data file
DATA_PATH = './acii-2025/grouped_by_user_id_sorted_by_timestamp.json'

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_sessions(messages, time_key='timestamp', time_format='%Y-%m-%d %H:%M:%S.%f+00', session_gap=60*60):
    """Split a list of messages into sessions based on inactivity gap (in seconds)."""
    sessions = []
    current_session = []
    last_time = None
    for msg in messages:
        try:
            t = datetime.strptime(msg[time_key], time_format)
        except (KeyError, ValueError):
            continue  # skip messages with missing or malformed timestamps
        if last_time is None or (t - last_time).total_seconds() <= session_gap:
            current_session.append(msg)
        else:
            if current_session:
                sessions.append(current_session)
            current_session = [msg]
        last_time = t
    if current_session:
        sessions.append(current_session)
    return sessions

def count_code_snippets(text):
    # Count code blocks for a list of common languages and generic code blocks
    languages = [
        '', 'python', 'java', 'javascript', 'js', 'c++', 'cpp', 'c', 'bash', 'shell', 'json', 'yaml', 'html', 'css', 'markdown', 'r', 'go', 'rust', 'typescript', 'ts', 'sql', 'swift', 'kotlin', 'scala', 'perl', 'php', 'dart', 'matlab', 'objective-c', 'ocaml', 'lua', 'powershell', 'sh', 'makefile', 'dockerfile', 'ini', 'toml', 'latex', 'vb', 'groovy', 'haskell', 'fortran', 'assembly', 'asm', 'clojure', 'elixir', 'erlang', 'fsharp', 'julia', 'lisp', 'prolog', 'scheme', 'solidity', 'verilog', 'vhdl', 'zig', 'nim', 'coffeescript', 'elm', 'purescript', 'reason', 'sml', 'vala', 'crystal', 'idris', 'ada', 'apex', 'awk', 'basic', 'brainfuck', 'cobol', 'd', 'delphi', 'elm', 'factor', 'forth', 'foxpro', 'gdscript', 'haxe', 'j', 'labview', 'maxscript', 'mercury', 'nimrod', 'opencl', 'pascal', 'postscript', 'rexx', 'sas', 'smalltalk', 'tcl', 'vba', 'visualbasic', 'xquery', 'yacc', 'zsh'
    ]
    count = 0
    for lang in languages:
        if lang:
            count += text.count(f'```{lang}')
        else:
            count += text.count('```')
    return count

def main():
    data = load_data(DATA_PATH)
    participants = len(data)
    all_sessions = []
    turns_per_session = []
    tokens_per_turn = []
    all_timestamps = []

    time_format = '%Y-%m-%dT%H:%M:%S.%f%z'

    for user_id, messages in data.items():
        # Assume messages are sorted by timestamp
        sessions = extract_sessions(messages, time_format=time_format)
        all_sessions.extend(sessions)
        for session in sessions:
            turns_per_session.append(len(session))
            for msg in session:
                text = msg.get('content', '')
                tokens = len(text.split()) if text else 0
                tokens_per_turn.append(tokens)
                # Collect timestamps for conversation span
                ts = msg.get('timestamp')
                if ts:
                    try:
                        datetime.strptime(ts, time_format)
                        all_timestamps.append(ts)
                    except ValueError:
                        continue

    # Compute statistics
    def median_iqr(arr):
        arr = np.array(arr)
        median = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        return median, q1, q3

    if turns_per_session:
        turns_median, turns_q1, turns_q3 = median_iqr(turns_per_session)
        print(f"Turns per session: {turns_median:.1f} ({turns_q1:.1f}–{turns_q3:.1f})")
    else:
        print("Warning: No session data found for turns per session.")

    if tokens_per_turn:
        tokens_median, tokens_q1, tokens_q3 = median_iqr(tokens_per_turn)
        print(f"Tokens per turn: {tokens_median:.0f} ({tokens_q1:.0f}–{tokens_q3:.0f})")
    else:
        print("Warning: No data found for tokens per turn.")

    # Conversation span in days (overall)
    if all_timestamps:
        all_times = [datetime.strptime(ts, time_format) for ts in all_timestamps]
        span_days = (max(all_times) - min(all_times)).days
    else:
        span_days = 0

    # Per-participant conversation span (in days)
    participant_spans = []
    participant_num_sessions = []
    participant_total_turns = []
    participant_total_tokens = []
    participant_avg_tokens_per_turn = []
    participant_code_snippets = []
    participant_avg_turns_per_session = []
    participant_avg_session_duration = []  # in minutes
    participant_first_activity = []
    participant_last_activity = []
    participant_days_active = []
    # For response time, we need to know who is user/bot; skip if not available
    # participant_avg_response_time = []

    # --- Add: Per-participant code snippets per turn ---
    participant_code_snippets_per_turn = []

    for user_id, messages in data.items():
        user_timestamps = [msg.get('timestamp') for msg in messages if msg.get('timestamp')]
        user_times = []
        for ts in user_timestamps:
            try:
                user_times.append(datetime.strptime(ts, time_format))
            except ValueError:
                continue
        if user_times:
            user_span = (max(user_times) - min(user_times)).days
            participant_spans.append(user_span)
            participant_first_activity.append(min(user_times))
            participant_last_activity.append(max(user_times))
            days_active = len(set([dt.date() for dt in user_times]))
            participant_days_active.append(days_active)
        # Calculate number of sessions for this participant
        sessions = extract_sessions(messages, time_format=time_format)
        participant_num_sessions.append(len(sessions))
        # Turns
        num_turns = len(messages)
        participant_total_turns.append(num_turns)
        # Tokens and code snippets
        total_tokens = 0
        code_snippets = 0
        for msg in messages:
            text = msg.get('content', '')
            tokens = len(text.split()) if text else 0
            total_tokens += tokens
            code_snippets += count_code_snippets(text)
        participant_total_tokens.append(total_tokens)
        participant_code_snippets.append(code_snippets)
        # Avg tokens per turn
        avg_tokens = total_tokens / num_turns if num_turns > 0 else 0
        participant_avg_tokens_per_turn.append(avg_tokens)
        # Avg turns per session
        avg_turns_per_session = num_turns / len(sessions) if sessions else 0
        participant_avg_turns_per_session.append(avg_turns_per_session)
        # Avg session duration (in minutes)
        session_durations = []
        for session in sessions:
            if len(session) >= 2:
                try:
                    t0 = datetime.strptime(session[0]['timestamp'], time_format)
                    t1 = datetime.strptime(session[-1]['timestamp'], time_format)
                    duration = (t1 - t0).total_seconds() / 60.0
                    session_durations.append(duration)
                except Exception:
                    continue
        avg_session_duration = np.mean(session_durations) if session_durations else 0
        participant_avg_session_duration.append(avg_session_duration)
        # --- Add: code snippets per turn for this participant ---
        code_snippets_per_turn = code_snippets / num_turns if num_turns > 0 else 0
        participant_code_snippets_per_turn.append(code_snippets_per_turn)

    # --- Per-participant statistics ---
    print("\n===== Per-participant statistics =====")
    def print_stat(name, arr, fmt='.1f'):
        if arr:
            median, q1, q3 = median_iqr(arr)
            print(f"{name}: {format(median, fmt)} ({format(q1, fmt)}–{format(q3, fmt)})")
        else:
            print(f"Warning: No data found for {name}.")

    print_stat("Per-participant conversation span (days)", participant_spans)
    print_stat("Per-participant number of sessions", participant_num_sessions)
    print_stat("Per-participant total turns", participant_total_turns, fmt='.0f')
    print_stat("Per-participant total tokens", participant_total_tokens, fmt='.0f')
    print_stat("Per-participant avg tokens per turn", participant_avg_tokens_per_turn, fmt='.1f')
    print_stat("Per-participant code snippets sent", participant_code_snippets, fmt='.0f')
    print_stat("Per-participant avg turns per session", participant_avg_turns_per_session, fmt='.1f')
    print_stat("Per-participant avg session duration (min)", participant_avg_session_duration, fmt='.1f')
    # --- Add: print per-participant code snippets per turn ---
    print_stat("Per-participant code snippets per turn", participant_code_snippets_per_turn, fmt='.3f')
    # First and last activity (print range)
    if participant_first_activity and participant_last_activity:
        first = min(participant_first_activity)
        last = max(participant_last_activity)
        print(f"First activity: {first.strftime('%Y-%m-%d')}, Last activity: {last.strftime('%Y-%m-%d')}")
    else:
        print("Warning: No data for first/last activity.")
    print_stat("Per-participant days active", participant_days_active, fmt='.0f')
    # Response time skipped (need user/bot roles)

    # --- Overall statistics ---
    print("\n===== Overall statistics =====")
    print(f"Participants: {participants}")
    print(f"Sessions: {len(all_sessions)}")
    print(f"Conversation span (days): {span_days}")

    # Total tokens
    total_tokens = sum(participant_total_tokens)
    print(f"Tokens: {total_tokens}")

    # Total code snippets
    total_code_snippets = sum(participant_code_snippets)
    print(f"Code snippets: {total_code_snippets}")

    # --- Add: overall code snippets per turn ---
    total_turns = sum(participant_total_turns)
    overall_code_snippets_per_turn = total_code_snippets / total_turns if total_turns > 0 else 0
    print(f"Code snippets per turn: {overall_code_snippets_per_turn:.3f}")

    # Total session duration (min)
    total_session_duration = 0
    for session in all_sessions:
        if len(session) >= 2:
            try:
                t0 = datetime.strptime(session[0]['timestamp'], time_format)
                t1 = datetime.strptime(session[-1]['timestamp'], time_format)
                duration = (t1 - t0).total_seconds() / 60.0
                total_session_duration += duration
            except Exception:
                continue
    print(f"Session duration (min): {total_session_duration:.1f}")

    # Total days active (unique days with at least one message)
    all_dates = set()
    for ts in all_timestamps:
        try:
            dt = datetime.strptime(ts, time_format)
            all_dates.add(dt.date())
        except Exception:
            continue
    print(f"Days active: {len(all_dates)}")

if __name__ == "__main__":
    main()