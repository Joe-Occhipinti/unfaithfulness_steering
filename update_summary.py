
import json
import os

SUMMARY_FILE = "data/sprint_5_2025-11-15/steered/summary_gradient_2025-12-01_shard_1.json"

def update_summary():
    try:
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            
        print("Loaded summary file.")
        
        # 1. Update Metadata
        if 'metadata' in summary:
            # Update layers_tested
            if 'layers_tested' in summary['metadata']:
                original_layers = summary['metadata']['layers_tested']
                summary['metadata']['layers_tested'] = [l for l in original_layers if l != 28]
                print(f"Updated layers_tested: {summary['metadata']['layers_tested']}")
                
            # Update target_values (no change needed for shard 1 as 10 is likely not there or handled, but safe to keep logic)
            if 'target_values' in summary['metadata']:
                original_targets = summary['metadata']['target_values']
                summary['metadata']['target_values'] = [t for t in original_targets if t != 10]
                print(f"Updated target_values: {summary['metadata']['target_values']}")
        
        # 2. Filter Configurations
        if 'configurations' in summary:
            original_configs = summary['configurations']
            new_configs = {}
            removed_count = 0
            
            for key, value in original_configs.items():
                # Key format: layer_8_offensive_target_1
                parts = key.split('_')
                
                try:
                    layer = int(parts[1])
                    target = int(parts[-1])
                    
                    if layer == 28:
                        removed_count += 1
                        continue
                    
                    new_configs[key] = value
                except (ValueError, IndexError):
                    # Keep if we can't parse (safety)
                    new_configs[key] = value
            
            summary['configurations'] = new_configs
            print(f"Removed {removed_count} configurations.")
            print(f"Remaining configurations: {len(new_configs)}")
            
        # Save updated summary
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Updated summary saved to {SUMMARY_FILE}")
        
    except FileNotFoundError:
        print(f"File not found: {SUMMARY_FILE}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_summary()
