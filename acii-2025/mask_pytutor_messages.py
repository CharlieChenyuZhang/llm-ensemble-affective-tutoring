#!/usr/bin/env python3
"""
Script to mask contents in pytutor_chat_messages.json
Replaces the 'content' field with "MASKED_DATA" for all objects.
"""

import json
import os
from datetime import datetime

def main():
    """Main function to mask contents in the pytutor_chat_messages.json file."""
    
    # File path
    input_file = "grouped_by_user_id_sorted_by_timestamp.json"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found in current directory")
        print("Please make sure you're running this script from the acii-2025 directory")
        return
    
    print(f"Processing file: {input_file}")
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"pytutor_chat_messages_backup_{timestamp}.json"
    
    try:
        # Load the JSON data
        print("Loading JSON file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} objects from JSON file")
        
        # Create backup
        print(f"Creating backup: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Backup created successfully")
        
        # Mask contents for each object
        modified_count = 0
        for obj in data:
            if 'content' in obj:
                obj['content'] = "MASKED_DATA"
                modified_count += 1
        
        print(f"Masked contents from {modified_count} objects")
        
        # Save the modified data back to the original file
        print("Saving modified data...")
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully updated {input_file}")
        print(f"Backup saved as: {backup_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
