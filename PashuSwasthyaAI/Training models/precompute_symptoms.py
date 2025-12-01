import json
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import os

# ---------------------------
# Load dataset
# ---------------------------
with open("Dataset/veterinary_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
records = data["records"]

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ---------------------------
# Precompute embeddings
# ---------------------------
if os.path.exists("precomputed_symptoms.npz"):
    print("precomputed_symptoms.npz already exists. Delete if you want to regenerate.")
    exit()

print("⚙ Generating precomputed_symptoms.npz ...")

all_symptoms = []
all_embeddings = []

for r in records:
    # Record-level embedding
    if "Symptom_Embeddings" not in r:
        r["Symptom_Embeddings"] = embedding_model.encode(r["Symptoms"], convert_to_numpy=True)
    # Global symptoms
    for s in r["Symptoms"]:
        all_symptoms.append(s)
        emb = embedding_model.encode([s], convert_to_numpy=True)[0]
        all_embeddings.append(emb)

all_embeddings = np.vstack(all_embeddings)

# Save
np.savez_compressed(
    "precomputed_symptoms.npz",
    records=records,
    all_symptoms=all_symptoms,
    all_symptom_embeddings=all_embeddings
)

print(f"✅ Created precomputed_symptoms.npz with {len(all_symptoms)} symptoms.")
