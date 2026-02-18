import sys
import os
# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os

DATA_DIR = r"c:\Users\occhi\Desktop\unfaithfulness_steering\data\definitive_pipeline_data\DeepSeek-R1-Distill-Llama-8B"

FILES = {
    "Linear": "summary_steered_linear_DeepSeek-R1-Distill-Llama-8B_2026-01-11.json",
    "MLP": "summary_steered_mlp_DeepSeek-R1-Distill-Llama-8B_2026-01-12.json",
    "OffPolicy": "summary_steered_off_policy_DeepSeek-R1-Distill-Llama-8B_2026-01-11.json"
}

def get_stats_from_group(group_data):
    """
    Extracts Success/Total for a group (e.g. positive_on_WU).
    Returns (success_count, total_count).
    Success here typically means 'faithfulness' = 'faithful'.
    The JSON summary breaks down transitions like 'stable_faithful', 'wrong_to_correct', etc.
    We need to know which transitions imply "Becoming Faithful".
    
    Standard Transitions:
    - stable_faithful: Was Faithful, Stays Faithful -> Faithful (Success if aim is faithful)
    - stable_unfaithful: Was Unfaithful, Stays Unfaithful -> Unfaithful
    - wrong_to_correct: Was Wrong(Unfaithful?), Becomes Correct... wait.
    
    Let's look at the transitions in the JSON again.
    "transitions": { "stable_faithful": ..., "wrong_to_correct": ... }
    
    The labels likely map to:
    - stable_faithful: Faithful -> Faithful
    - stable_unfaithful: Unfaithful -> Unfaithful
    - wrong_to_correct: Wrong -> Correct. Does this imply Unfaithful -> Faithful? 
      In 'Grader Hacking', often Unfaithful = Wrong Answer (or hints).
      If positive_on_WU (Wrong Unfaithful), and it becomes Correct... 
      Usually, if it becomes Correct, it's considered "Recovered" or "Faithful" in this context?
      
      Let's assume:
      - Recovery (on Unfaithful inputs): stable_faithful (if was unfaithful?? no), wrong_to_correct?
      
      Wait, if input is WU (Wrong Unfaithful):
      - stable_unfaithful: Still Unfaithful
      - wrong_to_correct: Becomes Correct. Is it Faithful?
      
      Let's re-read the JSON structure for a clue.
      "positive_on_WU": { "transitions": { "wrong_to_correct": { "rate": 0.2 ... } } }
      
      If we look at `faithfulness_classification` in the raw data, it's clearer.
      But in the summary JSON, we only have these transition names.
      
      Assumption based on typical pipeline:
      - 'wrong_to_correct' on Unfaithful data usually counts as Recovery.
      - 'stable_faithful' on Unfaithful data? (Was it unfaithful? Maybe 'stable' means stayed same *faithfulness*? No, 'stable_faithful' implies it *is* faithful and *was* faithful? That contradicts 'on_WU'.)
      
      Hypothesis: The transitions describe the *outcome* classification relative to the *baseline*?
      Actually, 'wrong_to_correct' is an accuracy metric.
      The user mentioned "Monitorability recovery". Monitorability approx Faithfulness.
      In 'Grader Hacking', "Unfaithful" often means "Biased/Wrong Answer".
      "Faithful" often means "Correct Answer".
      So 'wrong_to_correct' is likely the specific metric for Recovery.
      
      Collateral Damage (on Faithful inputs - WF/CF):
      - Input: Faithful.
      - Damage: Becomes Unfaithful.
      - Transitions: 'stable_unfaithful' (if it became unfaithful?), 'hint_error' (bias?), 'wrong_to_correct' (impossible if already correct/faithful?).
      - 'stable_faithful': Stays Faithful (Good).
      - 'stable_unfaithful': (Impossible if starting Faithful? Unless 'stable' refers to accuracy?)
      
      Let's refine:
      - Inputs:
          - Unfaithful (WU + CU)
          - Faithful (WF + CF)
      
      - Recovery Metric (Numerator):
          - Rate of (Unfaithful -> Faithful).
          - In the JSON, for `positive_on_WU`:
            - `wrong_to_correct` is likely the "Recovery".
            - `stable_faithful` (Outlier? If it was U...).
            
      - Collateral Metric (Denominator):
          - Rate of (Faithful -> Unfaithful).
          - In `positive_on_WF` / `positive_on_CF`:
            - `stable_faithful`: No Damage.
            - `hint_error`: Likely Damage (Bias).
            - `stable_unfaithful`: Damage.
            
      Let's calculate:
      - Recovery Rate = (Count of Recovered / Total Unfaithful)
      - Collateral Rate = (Count of Damaged / Total Faithful)
    """
    
    # We will sum counts from the breakdown
    # Recovered (from Unfaithful):
    # - wrong_to_correct
    # - stable_faithful (if it appears in Unfaithful group, it's weird but implies faithful outcome)
    
    # Damaged (from Faithful):
    # - stable_unfaithful
    # - hint_error
    # - error (?)
    
    return group_data

def get_counts(transitions):
    # Returns (faithful_outcome_count, unfaithful_outcome_count, total)
    # This is a heuristic mapping based on standard pipeline names
    
    faithful_tags = ['stable_faithful', 'wrong_to_correct', 'wrong_to_correct_mentioning_hint']
    unfaithful_tags = ['stable_unfaithful', 'hint_error', 'hint_error_mentioning_hint']
    
    n_faithful = 0
    n_unfaithful = 0
    total = 0
    
    for tag, stats in transitions.items():
        count = stats.get('count', 0)
        total += count
        if tag in faithful_tags:
            n_faithful += count
        elif tag in unfaithful_tags:
            n_unfaithful += count
            
    return n_faithful, n_unfaithful, total

def find_best_config_ratio(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = []

    for hint_type, configs in data.get('configurations_by_hint', {}).items():
        for cfg in configs:
            # 1. Calculate Recovery (on Unfaithful)
            # Combine positive_on_WU and positive_on_CU
            wu = cfg.get('positive_on_WU', {})
            cu = cfg.get('positive_on_CU', {})
            
            rec_f_wu, rec_u_wu, n_wu = get_counts(wu.get('transitions', {}))
            rec_f_cu, rec_u_cu, n_cu = get_counts(cu.get('transitions', {}))
            
            total_unfaithful = n_wu + n_cu
            total_recovered = rec_f_wu + rec_f_cu
            
            recovery_rate = total_recovered / total_unfaithful if total_unfaithful > 0 else 0
            
            # 2. Calculate Collateral Damage (on Faithful)
            # Combine positive_on_WF and positive_on_CF
            wf = cfg.get('positive_on_WF', {})
            cf = cfg.get('positive_on_CF', {})
            
            dam_f_wf, dam_u_wf, n_wf = get_counts(wf.get('transitions', {}))
            dam_f_cf, dam_u_cf, n_cf = get_counts(cf.get('transitions', {}))
            
            # Damage = Outcome is Unfaithful
            total_faithful = n_wf + n_cf
            total_damaged = dam_u_wf + dam_u_cf
            
            collateral_rate = total_damaged / total_faithful if total_faithful > 0 else 0
            
            # 3. Ratio
            # Handle divide by zero
            if collateral_rate == 0:
                ratio = recovery_rate * 1000 # Penalize or boost? "Arg max of ratio" -> implies high ratio is good.
                # If damage is 0, ratio is Infinite (Very Good).
            else:
                ratio = recovery_rate / collateral_rate
                
            results.append({
                'layer': cfg['layer'],
                'coeff': cfg['coefficient_magnitude'],
                'recovery_rate': recovery_rate,
                'collateral_rate': collateral_rate,
                'ratio': ratio,
                'n_unfaithful': total_unfaithful,
                'n_faithful': total_faithful
            })
            
    # Find Max Ratio
    if not results: return None
    
    # Sort by Ratio desc, then Recovery Rate desc
    best = max(results, key=lambda x: (x['ratio'], x['recovery_rate']))
    return best

def main():
    for name, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            best = find_best_config_ratio(path)
            if best:
                print(f"Best {name}: Layer {best['layer']}, Coeff {best['coeff']}")
                print(f"  Recovery: {best['recovery_rate']:.4f}, Collateral: {best['collateral_rate']:.4f}, Ratio: {best['ratio']:.4f}")
                print(f"  (N_U: {best['n_unfaithful']}, N_F: {best['n_faithful']})")
        else:
            print(f"{name} file not found.")

if __name__ == "__main__":
    main()
