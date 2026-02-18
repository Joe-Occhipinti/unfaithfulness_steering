import argparse
import sys
import os
import subprocess
from pathlib import Path

# Mapping of stage names to script files
SCRIPTS = {
    "baseline": "scripts/eval_baseline.py",
    "annotate": "scripts/annotate_faithfulness.py",
    "faithfulness": "scripts/eval_faithfulness.py",
    "hinted": "scripts/eval_hinted.py",
    "steering": "scripts/eval_steering.py",
    "extract": "scripts/extract_activations.py",
    "probes": "scripts/train_probes.py",
    "vectors": "scripts/generate_steering_vectors.py",
    "stats": "scripts/statistical_analysis.py",
    "process": "scripts/process_answers.py",
    "plot": "scripts/plot_variations.py",
    "steered_faithfulness": "scripts/eval_faithfulness_steered.py"
}


def main():
    parser = argparse.ArgumentParser(
        description="Unfaithfulness Steering Pipeline Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage baseline --models Qwen/Qwen3-32B --subjects high_school_biology
  python main.py --stage train_probes --model Qwen/Qwen3-32B
  python main.py --stage steering --model Qwen/Qwen3-32B --layer 15
        """
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=SCRIPTS.keys(),
        help="Pipeline stage to execute"
    )
    
    # Parse known args to get the stage, pass the rest to the script
    args, unknown_args = parser.parse_known_args()
    
    script_rel_path = SCRIPTS[args.stage]
    script_path = Path(__file__).parent / script_rel_path
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
        
    print(f"=== Running Stage: {args.stage} ===")
    print(f"Script: {script_rel_path}")
    print(f"Arguments: {unknown_args}")
    print("=" * 60)
    
    # Construct command
    # Use config-aware python execution if needed, but simple subprocess is usually fine
    cmd = [sys.executable, str(script_path)] + unknown_args
    
    try:
        # Run the script and wait for it to complete
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error execution script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
