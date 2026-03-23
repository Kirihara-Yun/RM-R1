import json
import os
from chat_prompt_chinese import SYSTEM_PROMPT_CHAT_CHINESE as SYSTEM_PROMPT

# --- Configuration ---
# Input files (relative to this script's location)
TRAIN_FILE_IN = 'train.jsonl'
TEST_FILE_IN = 'test.jsonl'

# Output files will be created in the same directory
TRAIN_FILE_OUT = 'train_with_sys.jsonl'
TEST_FILE_OUT = 'test_with_sys.jsonl'
# --- End Configuration ---


def inject_system_prompt(input_path: str, output_path: str):
    """
    Reads a jsonl file, prepends the SYSTEM_PROMPT to each record's
    'context_messages' if it's not already present, and writes the result
    to a new file. The original file is not modified.
    """
    print(f"Processing '{input_path}' -> '{output_path}'...")
    
    # Check if the input file exists before proceeding
    if not os.path.exists(input_path):
        print(f"  [Warning] Input file not found: '{input_path}'. Skipping.")
        return 0

    records_processed = 0
    records_skipped = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [Warning] Line {i+1}: Skipping malformed JSON in '{input_path}'.")
                continue

            # Ensure 'context_messages' exists and is a list
            messages = record.get("context_messages", [])
            if not isinstance(messages, list):
                 print(f"  [Warning] Line {i+1}: 'context_messages' is not a list. Skipping record.")
                 records_skipped += 1
                 continue

            # Prepend system prompt only if the first message is not already 'system'
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            
            record["context_messages"] = messages
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_processed += 1
            
    print(f"  [Info] Done. Processed {records_processed} records.")
    if records_skipped > 0:
        print(f"  [Info] Skipped {records_skipped} records due to formatting issues.")
    return records_processed


if __name__ == "__main__":
    # Get the directory where the script is located to build absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_in_path = os.path.join(script_dir, TRAIN_FILE_IN)
    test_in_path = os.path.join(script_dir, TEST_FILE_IN)
    train_out_path = os.path.join(script_dir, TRAIN_FILE_OUT)
    test_out_path = os.path.join(script_dir, TEST_FILE_OUT)

    print("--- Starting Data Preprocessing: Inject System Prompt ---")
    
    inject_system_prompt(train_in_path, train_out_path)
    inject_system_prompt(test_in_path, test_out_path)
    
    print("\n--- Preprocessing Complete ---")
    print(f"New training file created: {TRAIN_FILE_OUT}")
    print(f"New testing file created:  {TEST_FILE_OUT}")
    print("\nReminder: Please update your training script (e.g., train_rm_r1_rlvr_*.sh) to point to these new '_with_sys.jsonl' files.")
