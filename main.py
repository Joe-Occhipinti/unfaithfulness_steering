import argparse
import sys
import subprocess
from pathlib import Path

# Mapping of stage names to script files
SCRIPTS = {
    "baseline": "scripts/eval_baseline.py",
    "hinted": "scripts/eval_hinted.py",
    "annotate": "scripts/annotate_faithfulness.py",
    "extract": "scripts/extract_activations.py",
    "vectors": "scripts/generate_steering_vectors.py",
    "probes": "scripts/train_probes.py",
    "steering": "scripts/eval_steering.py",
    "steered_faithfulness": "scripts/eval_faithfulness_steered.py",
    "stats": "scripts/statistical_analysis.py",
    "process": "scripts/process_answers.py",
    "plot": "scripts/plot_variations.py",
    "faithfulness": "scripts/eval_faithfulness.py"
}

DEFAULT_MODELS = [
    "Qwen3-32B",
    # Add other default models here
]

def run_script(stage, args, ignore_failure=False):
    """Run a single script for a given stage with arguments."""
    script_rel_path = SCRIPTS.get(stage)
    if not script_rel_path:
        print(f"Error: Unknown stage '{stage}'")
        return False
        
    script_path = Path(__file__).parent / script_rel_path
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        return False
        
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n[MAIN] Running stage '{stage}' -> {script_path.name}")
    print(f"[MAIN] Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[MAIN] Stage '{stage}' failed with exit code {e.returncode}")
        if not ignore_failure:
            return False
        return True # Ignored

def run_full_pipeline(models, extra_args):
    """Run the full pipeline for a set of models."""
    print(f"\n{'='*80}")
    print(f"STARTING FULL PIPELINE FOR MODELS: {models}")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\n{'>'*40}")
        print(f"PROCESSING MODEL: {model}")
        print(f"{'<'*40}\n")
        
        # NOTE: process_answers.py uses --model, others use --model_name
        # We also need to be careful not to pass extra_args (like --subjects) to scripts that don't support them
        # if those scripts use strict argument parsing. 
        # For now, we pass extra_args to evaluation scripts, but NOT to process_answers to be safe.
        
        # 1. Baseline Evaluation
        if not run_script("baseline", ["--model_name", model] + extra_args): return
        if not run_script("process", ["--model", model, "--dataset-type", "baseline"]): return
        
        # 2. Hinted Evaluation
        if not run_script("hinted", ["--model_name", model] + extra_args): return
        if not run_script("process", ["--model", model, "--dataset-type", "hinted"]): return

        # 2a. Post-Hinted Faithfulness Evaluation (Global)
        if not run_script("faithfulness", ["--model_name", model] + extra_args): return
        
        # 3. Faithfulness Annotation (Local)
        # Note: annotate likely needs extra args if provided
        if not run_script("annotate", ["--model_name", model] + extra_args): return
        
        # 4. Activation Extraction
        if not run_script("extract", ["--model_name", model] + extra_args): return
        
        # 5. Steering Vectors
        if not run_script("vectors", ["--model_name", model] + extra_args): return
        
        # 6. Train Probes
        if not run_script("probes", ["--model_name", model] + extra_args): return
        
        # 7. Steering Evaluation
        if not run_script("steering", ["--model_name", model] + extra_args): return
        
        # 8. Process Steered Answers
        # We attempt to process standard 'steered' outputs. 
        # If the user ran steering with specific settings that produce different output types, 
        # this might warn.
        # process_answers.py might fail if files are missing, so we ignore failure here to let pipeline verify next steps.
        # But per user request, we SHOULD run it. 
        if not run_script("process", ["--model", model, "--dataset-type", "steered"], ignore_failure=True): 
             print("[MAIN] Warning: 'steered' processing failed. Checking for other types...")
             
        # 9. Steered Faithfulness
        if not run_script("steered_faithfulness", ["--model_name", model] + extra_args): return

    print("\n[MAIN] Full pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Unfaithfulness Steering Pipeline Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a specific stage
  python main.py --stage baseline --model_name Qwen3-32B

  # Run full pipeline for all default models
  python main.py --full_pipeline

  # Run full pipeline for specific models
  python main.py --full_pipeline --models Qwen3-32B DeepSeek-R1-Distill-Llama-8B --subjects biology
        """
    )
    
    parser.add_argument("--stage", choices=list(SCRIPTS.keys()), help="Pipeline stage to run")
    parser.add_argument("--full_pipeline", action="store_true", help="Run the entire pipeline sequentially")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to process in full pipeline mode")
    
    args, unknown_args = parser.parse_known_args()
    
    if args.full_pipeline:
        run_full_pipeline(args.models, unknown_args)
    elif args.stage:
        run_script(args.stage, unknown_args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
