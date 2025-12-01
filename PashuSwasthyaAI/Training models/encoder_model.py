import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sentence_transformers import SentenceTransformer, util
from google import genai  # Google Gemini SDK

# ---------------------------
# ✅ Set your Gemini API Key here
# ---------------------------
GOOGLE_API_KEY = "AIzaSyDhKETwyNhp1X0is-PSIrZ-yMthvEjJvXU"  # <--- Replace with your real key
client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------------------
# Load Dataset
# ---------------------------
with open("Dataset/veterinary_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["records"]

# ---------------------------
# Prepare ML Features
# ---------------------------
print("🔄 Preparing embeddings and ML features...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

X_symptoms = []
X_animal = []
y = []

animal_types = list(set([r["Animal_Type"] for r in records]))
animal_encoder = OneHotEncoder(sparse_output=False).fit(np.array(animal_types).reshape(-1, 1))
disease_encoder = LabelEncoder()
disease_encoder.fit([r["Disease"] for r in records])

for r in records:
    symptom_emb = model.encode(r["Symptoms"])
    avg_emb = np.mean(symptom_emb, axis=0)
    X_symptoms.append(avg_emb)
    animal_vec = animal_encoder.transform([[r["Animal_Type"]]])[0]
    X_animal.append(animal_vec)
    y.append(disease_encoder.transform([r["Disease"]])[0])

X = np.hstack([X_symptoms, X_animal])
y = np.array(y)

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Train ML Model
# ---------------------------
print("⚙ Training ML model (XGBoost)...")
clf = XGBClassifier(
    objective='multi:softprob',
    num_class=len(disease_encoder.classes_),
    eval_metric='mlogloss'
)
clf.fit(X_train, y_train)

# Save model & encoders
joblib.dump(clf, "xgb_vet_model.pkl")
joblib.dump(animal_encoder, "animal_encoder.pkl")
joblib.dump(disease_encoder, "disease_encoder.pkl")
print("✅ ML model trained and saved successfully!")
