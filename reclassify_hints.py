
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path if needed
sys.path.append(os.getcwd())

from src.data import load_jsonl, save_jsonl
from src.hint_mention import classify_hint_mentions
from src.steered_global_faithfulness import get_record_id

async def reclassify_file(input_path: Path):
    print(f"Loading {input_path}...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records.")
    
    # Identify records to process
    # We want to process records that are NOT stable.
    # In annotated files, these have rule_classification NOT IN ['stable_correct', 'stable_wrong']
    # OR we can just look for where we'd normally run hint detection.
    
    to_process_indices = []
    to_process_records = []
    
    skipped_count = 0
    
    for idx, record in enumerate(records):
        # reliable way: check if rule_classification indicates checks needed
        rule = record.get('rule_classification')
        
        # If rule is missing, we can't easily skip without re-implementing rule logic.
        # Assuming input is an ANNOTATED file.
        
        if rule in ['stable_correct', 'stable_wrong']:
            skipped_count += 1
            continue
            
        # Also skip if it's an error or weird state, but generally we process the rest
        # The relevant classes are usually: wrong_to_correct, hint_error, incomplete
        
        to_process_indices.append(idx)
        to_process_records.append(record)
        
    print(f"Found {len(to_process_records)} records to re-classify (skipped {skipped_count} stable/other records).")
    
    if not to_process_records:
        print("No records to process.")
        return

    # Run classification
    # Note: classify_hint_mentions uses the logic in src/hint_mention.py, which we fixed.
    print("Starting batch classification...")
    results = await classify_hint_mentions(
        records=to_process_records,
        verbose=True
    )
    
    # merging results
    updates_count = 0
    for local_idx, result in results.items():
        if result == 'error':
            continue
            
        global_idx = to_process_indices[local_idx]
        record = records[global_idx]
        
        # Update the field
        # Note: The previous run might have had False (default) or True (false positive)
        # We overwrite it.
        if record.get('hint_mentioned') != result:
            updates_count += 1
            
        record['hint_mentioned'] = result
        
    print(f"Updated {updates_count} records with new values.")
    
    # Save
    # We'll save to the same filename to update it, or maybe a suffix?
    # User said "re-run", implying update. Let's start with overwriting or suffix.
    # Safest is to save to new file then rename/user choice.
    # Let's save to the original path (overwrite) as that's usually what's expected for "re-run".
    # Actually, let's keep a backup just in case.
    
    backup_path = input_path.with_suffix('.jsonl.bak')
    if not backup_path.exists():
        print(f"Creating backup at {backup_path}")
        save_jsonl(records, backup_path) # Actually save_jsonl might not be efficient for just copying, but ok.
        
    print(f"Saving updated records to {input_path}...")
    save_jsonl(records, input_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Re-run hint mention classification on annotated files.")
    parser.add_argument('files', nargs='+', help="Input .jsonl files (annotated)")
    args = parser.parse_args()
    
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"File not found: {f}")
            continue
        
        asyncio.run(reclassify_file(path))

if __name__ == "__main__":
    main()
