from dotenv import load_dotenv
load_dotenv()

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Google Gemini
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# Import system prompt from constants
from constants import SYSTEM_PROMPT

# ========== CONFIGURATION ==========
INPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_llm_analysis.json")
OUTPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_gemini_analysis.json")

# ========== GEMINI SETUP ==========
# Set your Google API key in the environment variable GOOGLE_API_KEY
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ========== RATE LIMITER ==========
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = []  # timestamps of recent calls

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # Remove timestamps older than the period
                self.calls = [t for t in self.calls if now - t < self.period]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                else:
                    # Wait until the earliest call is outside the period
                    sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

# Global rate limiter: 1,000 RPM
rate_limiter = RateLimiter(max_calls=1000, period=60.0)

def call_gemini(text: str) -> List[Dict[str, Any]]:
    max_retries = 3
    for attempt in range(max_retries):
        rate_limiter.acquire()  # Enforce 1,000 RPM globally
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{SYSTEM_PROMPT}\n\nParagraph:\n{text}",
                # config=types.GenerateContentConfig(
                #     thinking_config=types.ThinkingConfig(thinking_budget=0)
                # ),
            )
            # Gemini returns a response object with .text containing the output
            completion = response.text.strip()
            # Try to extract the JSON array from the completion string
            start = completion.find('[')
            end = completion.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = completion[start:end+1]
                return json.loads(json_str)
            else:
                print(f"[Gemini] Could not find JSON array in completion: {completion}")
                return []
        except ClientError as e:
            if getattr(e, 'code', None) == 429:
                print(f"[Gemini] Rate limited (429). Waiting 1s before retrying... (attempt {attempt+1}/{max_retries})")
                time.sleep(1)
                continue
            else:
                print(f"[Gemini] ClientError: {e}")
                return []
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return []
    print(f"[Gemini] Failed after {max_retries} retries due to rate limiting.")
    return []

# ========== MAIN PROCESS ==========
def analyze_message(obj):
    text = obj.get("content", "")
    if not text.strip():
        obj["gemini_analysis"] = []
    else:
        obj["gemini_analysis"] = call_gemini(text)
    return obj

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_workers = 8
    write_lock = threading.Lock()
    num_items = len(data)
    completed = 0
    first_item = True

    # Open the output file and write the opening bracket
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        out_f.write('[\n')
        out_f.flush()

        def process_and_write(idx, obj):
            nonlocal first_item, completed
            result = analyze_message(obj)
            with write_lock:
                if not first_item:
                    out_f.write(',\n')
                else:
                    first_item = False
                json.dump(result, out_f, ensure_ascii=False)
                out_f.flush()
                completed += 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_write, idx, obj) for idx, obj in enumerate(data)]
            for _ in tqdm(as_completed(futures), total=num_items, desc="Analyzing messages with Gemini"):
                pass  # Progress bar only; all writing is done in process_and_write

        # After all threads finish, write the closing bracket
        out_f.write('\n]\n')
        out_f.flush()
    print(f"Gemini analysis complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 