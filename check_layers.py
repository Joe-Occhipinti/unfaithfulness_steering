import pickle

# Load the steering vectors
with open('data/sprint_6_2025-12-15/steering_vectors.pkl', 'rb') as f:
    data = pickle.load(f)

# Check layers
layers = sorted(data['steering_vectors'].keys())
print(f"Layers: {layers}")
print(f"Total layers: {len(data['steering_vectors'])}")
print(f"\nFirst layer: {layers[0]}")
print(f"Last layer: {layers[-1]}")
print(f"\nExpected: 0-31 (32 layers)")
print(f"Actual range: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

if layers == list(range(32)):
    print("\n✅ CORRECT: Layers are 0-31")
else:
    print("\n❌ INCORRECT: Layers are NOT 0-31!")
