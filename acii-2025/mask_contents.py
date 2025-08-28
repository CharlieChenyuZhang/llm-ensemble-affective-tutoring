#!/usr/bin/env python3
"""
Script to load pytutor_chat_messages_with_llm_analysis.json and mask contents
for each object by replacing the 'content' field with "MASKED_DATA".
"""

import json
import os
from datetime import datetime

def mask_contents_from_json(input_file, output_file=None):
    """
    Load JSON file and mask contents for each object.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional, defaults to input_file)
    """
    # Set output file to input file if not specified
    if output_file is None:
        output_file = input_file
    
    print(f"Loading JSON file: {input_file}")
    
    try:
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} objects from JSON file")
        
        # Mask contents for each object
        modified_count = 0
        for obj in data:
            if 'content' in obj:
                obj['content'] = "MASKED_DATA"
                modified_count += 1
        
        print(f"Masked contents from {modified_count} objects")
        
        # Save the modified data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved modified data to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function to run the script."""
    # Define the input file path
    input_file = "pytutor_chat_messages_with_llm_analysis.json"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found in current directory")
        print("Please make sure you're running this script from the acii-2025 directory")
        return
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"pytutor_chat_messages_with_llm_analysis_backup_{timestamp}.json"
    print(f"Creating backup: {backup_file}")
    
    try:
        # Copy original file to backup
        with open(input_file, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print("Backup created successfully")
    except Exception as e:
        print(f"Warning: Could not create backup - {e}")
    
    # Mask contents from the original file
    mask_contents_from_json(input_file)

if __name__ == "__main__":
    main()

