import json
from pathlib import Path
from collections import defaultdict
import ijson
import decimal

# INPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_llm_analysis.json")
# OUTPUT_PATH = Path("acii-2025/grouped_by_user_id_sorted_by_timestamp.json")

INPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_ensemble.json")
OUTPUT_PATH = Path("acii-2025/grouped_by_user_id_sorted_by_timestamp_ensemble.json")

def stream_json_array(file_path):
    """
    Efficiently stream a large JSON array file, yielding one object at a time using ijson.
    Handles multi-line objects and nested structures.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for obj in ijson.items(f, 'item'):
            yield obj

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def main():
    grouped = defaultdict(list)
    for obj in stream_json_array(INPUT_PATH):
        user_id = obj.get('user_id')
        if user_id is not None:
            grouped[user_id].append(obj)
        else:
            print(f"Warning: object without user_id: {obj}")
    # Sort each group by timestamp
    for user_id in grouped:
        grouped[user_id].sort(key=lambda x: x.get('timestamp', 0))
    # Save grouped result
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2, default=decimal_default)
    print(f"Grouped data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 