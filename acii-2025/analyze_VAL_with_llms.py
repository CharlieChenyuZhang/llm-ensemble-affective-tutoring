from dotenv import load_dotenv
load_dotenv()

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# AWS Bedrock
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# OpenAI
import openai
from pydantic import BaseModel, conint
from openai import OpenAI

# Import system prompt from constants
from constants import SYSTEM_PROMPT

# ========== CONFIGURATION ==========
INPUT_PATH = Path("acii-2025/pytutor_chat_messages.json") # NOTE: we cannot release the original data due to FERPA
OUTPUT_PATH = Path("acii-2025/pytutor_chat_messages_with_llm_analysis.json")

# ========== AWS BEDROCK SETUP ==========
# Set your AWS credentials in your environment or ~/.aws/credentials
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

def call_bedrock_claude(text: str) -> List[Dict[str, Any]]:
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    try:
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.2,
                "messages": [
                    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nParagraph:\n{text}"}
                ]
            }),
            contentType="application/json",
            accept="application/json"
        )
        raw_body = response["body"].read()
        result = json.loads(raw_body)
        # Extract the text from the Messages API response
        completion = result["content"][0]["text"].strip()
        # Try to extract the JSON array from the completion string
        start = completion.find('[')
        end = completion.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = completion[start:end+1]
            return json.loads(json_str)
        else:
            print(f"[Bedrock Claude] Could not find JSON array in completion: {completion}")
            return []
    except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError) as e:
        print(f"[Bedrock Claude] Error: {e}")
        return []

# ========== OPENAI GPT-4 SETUP ==========
# Set your OpenAI API key in the environment variable OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

client = OpenAI()

def call_openai_gpt4(text: str) -> List[dict]:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            max_tokens=2000,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        # Ensure the result is always a list
        if isinstance(result, dict):
            return [result]
        return result
    except Exception as e:
        print(f"[OpenAI GPT-4] Error: {e}")
        return []

# ========== MAIN PROCESS ==========
def analyze_message(obj):
    text = obj.get("content", "")
    if not text.strip():
        obj["claude_sonnet_analysis"] = []
        obj["gpt4_analysis"] = []
    else:
        obj["claude_sonnet_analysis"] = call_bedrock_claude(text)
        obj["gpt4_analysis"] = call_openai_gpt4(text)
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
            for _ in tqdm(as_completed(futures), total=num_items, desc="Analyzing messages with LLMs"):
                pass  # Progress bar only; all writing is done in process_and_write

        # After all threads finish, write the closing bracket
        out_f.write('\n]\n')
        out_f.flush()
    print(f"LLM analysis complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 