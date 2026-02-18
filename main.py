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
    "faithfulness": "scripts/eval_faithfulness.py",
    "generate_off_policy": "scripts/generate_off_policy_data.py"
}

def get_latest_file(pattern):
    import glob
    files = glob.glob(pattern)
    if not files: return None
    return max(files, key=os.path.getmtime)

def run_full_pipeline(models, extra_args):
    """Run the full pipeline for a set of models."""
    print(f"\n{'='*80}")
    print(f"STARTING FULL PIPELINE FOR MODELS: {models}")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\n{'>'*40}")
        print(f"PROCESSING MODEL: {model}")
        print(f"{'<'*40}\n")
        
        # 1. Baseline Evaluation
        if not run_script("baseline", ["--model_name", model] + extra_args): return
        if not run_script("process", ["--model", model, "--dataset-type", "baseline"]): return
        
        # 2. Hinted Evaluation
        if not run_script("hinted", ["--model_name", model] + extra_args): return
        if not run_script("process", ["--model", model, "--dataset-type", "hinted"]): return

        # 2a. Post-Hinted Faithfulness Evaluation (Global)
        if not run_script("faithfulness", ["--model_name", model] + extra_args): return
        
        # 3. Faithfulness Annotation (Local)
        if not run_script("annotate", ["--model_name", model] + extra_args): return

        # 3b. Generate Off-Policy Data (GLOBAL - Run Once)
        # We check if the global off-policy file exists. If not, we try to generate it using the current model's data.
        off_policy_file = "data/off_policy_responses.jsonl"
        
        if not os.path.exists(off_policy_file):
            # Try to find annotated data from the current model to use as seed
            annotated_pattern = f"data/{model}/behavioural/annotated_{model}_*.jsonl"
            annotated_file = get_latest_file(annotated_pattern)
            
            if annotated_file:
                print(f"\n[MAIN] Generating GLOBAL off-policy data from: {annotated_file}")
                # This will create data/off_policy_responses.jsonl
                if not run_script("generate_off_policy", ["--input_file", annotated_file, "--output_file", off_policy_file]): return
            else:
                 print(f"[MAIN] Warning: No annotated file found for {model} to generate off-policy data. Skipping extraction for this run.")
        else:
            print(f"\n[MAIN] Global off-policy data already exists at {off_policy_file}. Skipping generation.")

        # 4. Activation Extraction (Run for both On-Policy and Off-Policy modes)
        print(f"\n[MAIN] Extracting activations (Mode: On-Policy)")
        if not run_script("extract", ["--model", model, "--mode", "on-policy"] + extra_args): return
        
        # Only run off-policy extraction if the file exists
        if os.path.exists(off_policy_file):
            print(f"\n[MAIN] Extracting activations (Mode: Off-Policy)")
            # Note: extract_activations.py output location depends on the script logic. 
            # If off-policy-file is passed, it might try to put outputs next to it or in model dir. 
            # We updated extract_activations to put it in model's dir if found.
            if not run_script("extract", ["--model", model, "--mode", "off-policy", "--off-policy-file", off_policy_file] + extra_args): return
        
        # 5. Steering Vectors
        # Vectors need to be generated for both ON-POLICY and OFF-POLICY
        # The script defaults to on-policy. We should run both if off-policy exists.
        print(f"\n[MAIN] Generating Steering Vectors (Mode: On-Policy)")
        if not run_script("vectors", ["--model_name", model, "--mode", "on-policy"] + extra_args): return

        if os.path.exists(off_policy_file):
             print(f"\n[MAIN] Generating Steering Vectors (Mode: Off-Policy)")
             if not run_script("vectors", ["--model_name", model, "--mode", "off-policy"] + extra_args): return
        
        # 6. Train Probes
        
        # 6. Train Probes
        if not run_script("probes", ["--model_name", model] + extra_args): return
        
        # 7. Steering Evaluation
        if not run_script("steering", ["--model_name", model] + extra_args): return
        
        # 8. Process Steered Answers
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
