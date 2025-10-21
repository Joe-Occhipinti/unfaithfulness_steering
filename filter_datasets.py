import json
from pathlib import Path

# Input file path
input_file = Path("data/sprint4_2025-10-21/annotated/all_annotated_global_local_biased_econXsquare_histXmeta_psyXprof_2025-10-19.jsonl")
output_dir = Path("data/sprint4_2025-10-21/annotated")

# Define filter configurations
filters = [
    {
        "name": "black_square_pos_bas",
        "accuracy_label": "correct",
        "bias_label": "biased",
        "hint_template": "fewshot_black_square"
    },
    {
        "name": "black_square_neg_bas",
        "accuracy_label": "wrong",
        "bias_label": "biased",
        "hint_template": "fewshot_black_square"
    },
    {
        "name": "professor_pos_bas",
        "accuracy_label": "correct",
        "bias_label": "biased",
        "hint_template": "professor"
    },
    {
        "name": "professor_neg_bas",
        "accuracy_label": "wrong",
        "bias_label": "biased",
        "hint_template": "professor"
    },
    {
        "name": "metadata_pos_bas",
        "accuracy_label": "correct",
        "bias_label": "biased",
        "hint_template": "metadata"
    },
    {
        "name": "metadata_neg_bas",
        "accuracy_label": "wrong",
        "bias_label": "biased",
        "hint_template": "metadata"
    }
]

# Initialize counters for each filter
counters = {f["name"]: 0 for f in filters}

# Open output files
output_files = {
    f["name"]: open(output_dir / f"{f['name']}_2025-10-19.jsonl", 'w', encoding='utf-8')
    for f in filters
}

try:
    # Read and filter the input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            if line.strip():
                try:
                    record = json.loads(line)

                    # Check each filter
                    for filter_config in filters:
                        if (record.get("accuracy_label") == filter_config["accuracy_label"] and
                            record.get("bias_label") == filter_config["bias_label"] and
                            record.get("hint_template") == filter_config["hint_template"]):

                            output_files[filter_config["name"]].write(line)
                            counters[filter_config["name"]] += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")

            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines...")

finally:
    # Close all output files
    for f in output_files.values():
        f.close()

# Print summary
print("\n=== Filtering Complete ===")
for filter_config in filters:
    name = filter_config["name"]
    print(f"{name}: {counters[name]} records")
    print(f"  Output: {output_dir / f'{name}_2025-10-19.jsonl'}")
