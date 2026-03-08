#!/usr/bin/env python3
import json
import argparse
import sys
import os

def validate_jsonl(filepath):
    """
    Validates a JSONL file containing synthetic NER data.
    Ensures that every entity 'text' is an exact substring of the 'raw_text'.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    errors = 0
    valid_posts = 0
    total_entities = 0
    label_counts = {}

    print(f"Validating {filepath}...\n")

    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line: 
            continue
            
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {idx}: JSON Decode Error - {str(e)}")
            errors += 1
            continue

        raw_text = d.get("raw_text", "")
        post_id = d.get("id", f"line_{idx}")
        entities = d.get("entities", [])
        
        post_has_error = False
        
        for e in entities:
            entity_text = e.get("text", "")
            label = e.get("label", "UNKNOWN")
            
            # Count labels for stats
            label_counts[label] = label_counts.get(label, 0) + 1
            total_entities += 1

            # The exact substring check!
            if entity_text not in raw_text:
                print(f"❌ Error in [{post_id}]: Entity text '{entity_text}' not found in raw_text.")
                print(f"   Raw Text: '{raw_text}'\n")
                errors += 1
                post_has_error = True
                
        if not post_has_error:
            valid_posts += 1

    print("-" * 40)
    print("📋 VALIDATION SUMMARY")
    print("-" * 40)
    print(f"Total lines checked: {len([l for l in lines if l.strip()])}")
    print(f"Fully valid posts:   {valid_posts}")
    print(f"Total entities:      {total_entities}")
    print(f"Errors found:        {errors}")
    
    if label_counts:
        print("\n📊 Entity Breakdown:")
        for label, count in sorted(label_counts.items()):
            print(f"  - {label}: {count}")

    if errors > 0:
        print("\n⚠️  Validation FAILED. Please fix the exact substring errors above.")
        sys.exit(1)
    else:
        print("\n✅ Validation PASSED! All entities are exact substrings.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate exact substrings in NER JSONL data.")
    parser.add_argument(
        "--file", 
        type=str, 
        default="data/raw/synthetic.jsonl",
        help="Path to the JSONL file to validate (default: data/raw/synthetic.jsonl)"
    )
    args = parser.parse_args()
    
    validate_jsonl(args.file)
