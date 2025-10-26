# Debug cell to save dataset pickle file to Google Drive
# Run this in Colab after CELL 3 to save the dataset

import os
import pickle

DATASET_OUTPUT_FILE = "/content/drive/MyDrive/scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"

print(f"Attempting to save dataset to: {DATASET_OUTPUT_FILE}")

# Check if parent directory exists
parent_dir = os.path.dirname(DATASET_OUTPUT_FILE)
print(f"Parent directory: {parent_dir}")
print(f"Parent directory exists: {os.path.exists(parent_dir)}")

# Create parent directory if needed (though MyDrive should already exist)
if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)
    print(f"Created/verified parent directory")

# Try to save the dataset
try:
    print(f"\nSaving dataset with {len(dataset['activations'])} layers...")

    with open(DATASET_OUTPUT_FILE, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Dataset saved successfully!")

    # Verify it was saved
    if os.path.exists(DATASET_OUTPUT_FILE):
        file_size = os.path.getsize(DATASET_OUTPUT_FILE) / (1024**2)
        print(f"✅ File exists at {DATASET_OUTPUT_FILE}")
        print(f"   Size: {file_size:.2f} MB")
    else:
        print(f"⚠️ File was not found after saving!")

except Exception as e:
    print(f"❌ ERROR saving dataset: {e}")
    import traceback
    traceback.print_exc()

    # Try alternative location with subdirectory
    print(f"\nTrying alternative location with subdirectory...")
    ALT_DATASET_FILE = "/content/drive/MyDrive/datasets/scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"
    os.makedirs("/content/drive/MyDrive/datasets", exist_ok=True)

    with open(ALT_DATASET_FILE, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved to alternative location: {ALT_DATASET_FILE}")
    print(f"   Size: {os.path.getsize(ALT_DATASET_FILE) / (1024**2):.2f} MB")
