from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from xgboost import XGBClassifier
import joblib
import google.generativeai as genai
import re
import unicodedata
import string
from rapidfuzz import process, fuzz
from collections import Counter
from functools import lru_cache
import threading
import sys
import time
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ---------------------------
# Set Gemini API Key
# ---------------------------
GEMINI_API_KEY = "AIzaSyABMWswAPxrNsHsLrekuyqGHeXKJyY7Hoc"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# ---------------------------
# Load precomputed embeddings and dataset summaries
# ---------------------------
print("Loading data...")
data_npz = np.load("precomputed_symptoms.npz", allow_pickle=True)
records = data_npz["records"].tolist()
all_symptoms = data_npz["all_symptoms"].tolist()
all_symptom_embeddings = data_npz["all_symptom_embeddings"]

age_summary = {
    "Goat": {"min_age": 0.2, "max_age": 8.0, "avg_age": 4.08},
    "Cattle": {"min_age": 0.2, "max_age": 8.0, "avg_age": 4.1}
}

# Load ML models
clf = joblib.load("xgb_vet_model.pkl")
animal_encoder = joblib.load("animal_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Store active sessions with thread safety
sessions = {}
sessions_lock = threading.Lock()
SESSION_TIMEOUT = 3600  # 1 hour

# ---------------------------
# Session Management with Auto-cleanup
# ---------------------------
def create_session_id():
    """Generate unique session ID with timestamp"""
    return f"{int(time.time())}_{len(sessions)}"

def cleanup_old_sessions():
    """Remove expired sessions"""
    current_time = time.time()
    expired = []
    with sessions_lock:
        for sid, state in sessions.items():
            if current_time - state.get('created_at', 0) > SESSION_TIMEOUT:
                expired.append(sid)
        for sid in expired:
            del sessions[sid]
    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired sessions")

def get_session(session_id):
    """Thread-safe session retrieval"""
    with sessions_lock:
        return sessions.get(session_id)

def set_session(session_id, state):
    """Thread-safe session storage"""
    with sessions_lock:
        state['created_at'] = time.time()
        state['updated_at'] = time.time()
        sessions[session_id] = state

def delete_session(session_id):
    """Thread-safe session deletion"""
    with sessions_lock:
        if session_id in sessions:
            del sessions[session_id]
            print(f"🗑️ Deleted session: {session_id}")

# ---------------------------
# OPTIMIZATION 1: Pre-compute normalized symptoms
# ---------------------------
print("Pre-computing normalized data...")
normalized_symptoms_map = {}
symptom_to_original = {}

def precompute_symptom_mappings():
    """Pre-compute all symptom normalizations"""
    for symptom in all_symptoms:
        norm = normalize(symptom)
        normalized_symptoms_map[norm] = symptom
        symptom_to_original[norm] = symptom
    print(f"Pre-computed {len(normalized_symptoms_map)} symptom mappings")

# ---------------------------
# OPTIMIZATION 2: Pre-compute record data structures
# ---------------------------
def precompute_record_structures():
    """Pre-compute normalized versions of all records"""
    for record in records:
        record['_normalized_symptoms'] = [normalize(s) for s in record['Symptoms']]
        record['_normalized_animal'] = normalize(record['Animal_Type'])
        record['_normalized_subtype'] = normalize(record['Sub_Type'])
    print(f"Pre-computed structures for {len(records)} records")

# ---------------------------
# OPTIMIZATION 3: Create indexed lookup structures
# ---------------------------
animal_to_records = {}
disease_to_record = {}

def create_lookup_indexes():
    """Create fast lookup indexes"""
    global animal_to_records, disease_to_record
    
    for record in records:
        animal_key = (record['_normalized_animal'], record['_normalized_subtype'])
        if animal_key not in animal_to_records:
            animal_to_records[animal_key] = []
        animal_to_records[animal_key].append(record)
        
        disease_to_record[record['Disease']] = record
    
    print(f"Created indexes: {len(animal_to_records)} animal types, {len(disease_to_record)} diseases")

# ---------------------------
# Extract all animal types and subtypes from dataset
# ---------------------------
def extract_all_animal_subtypes_from_dataset(records):
    """Extract all unique animal types and subtypes from the dataset"""
    animal_mapping = {}
    for record in records:
        animal_type = record.get("Animal_Type", "")
        sub_type = record.get("Sub_Type", "")
        if animal_type:
            if animal_type not in animal_mapping:
                animal_mapping[animal_type] = set()
            if sub_type:
                animal_mapping[animal_type].add(sub_type)
    
    for key in animal_mapping:
        animal_mapping[key] = sorted(list(animal_mapping[key]))
    
    return animal_mapping

ANIMAL_SUBTYPES_MAP = extract_all_animal_subtypes_from_dataset(records)
print(f"Loaded animal types: {list(ANIMAL_SUBTYPES_MAP.keys())}")
for animal_type, subtypes in ANIMAL_SUBTYPES_MAP.items():
    print(f"  {animal_type}: {', '.join(subtypes)}")

# ---------------------------
# Utility: Normalize text (with caching)
# ---------------------------
@lru_cache(maxsize=10000)
def normalize(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    return text

def normalized_list(lst):
    return [normalize(x) for x in lst]

# ---------------------------
# LLM Cache with Expiration (Using Gemini)
# ---------------------------
llm_cache = {}
llm_cache_lock = threading.Lock()
CACHE_EXPIRY = 3600  # 1 hour

def cached_llm_call(cache_key, prompt, temperature=0.7, max_tokens=2048):
    """Cache LLM responses with expiration to avoid stale data"""
    current_time = time.time()
    
    with llm_cache_lock:
        if cache_key in llm_cache:
            cached_data = llm_cache[cache_key]
            if current_time - cached_data['timestamp'] < CACHE_EXPIRY:
                return cached_data['response']
            else:
                # Remove expired cache
                del llm_cache[cache_key]
    
    try:
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        response_text = response.text
        
        with llm_cache_lock:
            llm_cache[cache_key] = {
                'response': response_text,
                'timestamp': current_time
            }
        return response_text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return None

def clear_llm_cache():
    """Manually clear LLM cache"""
    with llm_cache_lock:
        llm_cache.clear()
    print("🧹 LLM cache cleared")

# ---------------------------
# Enhanced LLM-based animal detection
# ---------------------------
def extract_animal_type_with_llm(user_text):
    """Enhanced LLM-based animal detection with STRICT validation"""
    animal_type, sub_type = fallback_animal_detection(user_text)
    if animal_type and sub_type:
        return animal_type, sub_type
    
    animal_info = []
    for animal_type_key, subtypes in ANIMAL_SUBTYPES_MAP.items():
        animal_info.append(f"{animal_type_key}: {', '.join(subtypes)}")
    
    animal_list_str = "\n".join(animal_info)
    
    prompt = f"""
You are a multilingual veterinary assistant. Identify the animal from farmer's description.
CRITICAL: You MUST return ONLY animals from this EXACT list:
{animal_list_str}

Common translations:
- Cattle = गाय-बैल/पशु (Hindi) = गाई/गुरे (Marathi)
- Cow = गाय (Hindi/Marathi)
- Bull = बैल (Hindi/Marathi)
- Buffalo = भैंस (Hindi) = म्हशी (Marathi)
- Goat = बकरी (Hindi) = बकरी/शेळी (Marathi)
- Sheep = भेड़ (Hindi) = मेंढी (Marathi)

Farmer's text: "{user_text}"

STRICT RULES:
1. Animal_Type must EXACTLY match one from the list above (case-sensitive)
2. Sub_Type must EXACTLY match one of the valid subtypes but it cannot be case sensitive
3. NEVER return "Dog", "Cat", "Horse" or any animal NOT in the list
4. If goat mentioned → return {{"Animal_Type": "Goat", "Sub_Type": "Common"}}
5. If cow/cattle mentioned → return {{"Animal_Type": "Cattle", "Sub_Type": "Cow"}}

Return ONLY JSON: {{"Animal_Type": "Cattle", "Sub_Type": "Cow"}}
"""
    
    cache_key = f"animal_{hash(user_text)}"
    response_text = cached_llm_call(cache_key, prompt)
    
    if response_text:
        try:
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                animal_type = data.get("Animal_Type")
                sub_type = data.get("Sub_Type")
                
                if animal_type in ANIMAL_SUBTYPES_MAP:
                    if sub_type in ANIMAL_SUBTYPES_MAP[animal_type]:
                        print(f"✓ LLM Detected: {animal_type} - {sub_type}")
                        return animal_type, sub_type
                    else:
                        if ANIMAL_SUBTYPES_MAP[animal_type]:
                            fallback_subtype = ANIMAL_SUBTYPES_MAP[animal_type][0]
                            print(f"⚠ Subtype corrected: {sub_type} → {fallback_subtype}")
                            return animal_type, fallback_subtype
        except Exception as e:
            print(f"❌ LLM parsing failed: {e}")
    
    return None, None

def fallback_animal_detection(user_text):
    """Fallback with priority to exact matches"""
    text_lower = user_text.lower()
    
    animal_keywords = (
        (("Goat", "Common"), ("goat", "बकरी", "bakri", "bakra", "बकरा", "शेळी", "sheli", "शेळ", "shel")),
        (("Cattle", "Buffalo"), ("buffalo", "भैंस", "bhains", "म्हशी", "mhashi", "mahish")),
        (("Cattle", "Bull"), ("bull", "बैल", "bail", "साँड़", "sand")),
        (("Cattle", "Calf"), ("calf", "calves", "बछड़ा", "बछिया", "bachda", "bachiya", "वासरू", "vasaru")),
        (("Cattle", "Cow"), ("cow", "गाय", "gaay", "गाई", "gai")),
        (("Sheep", "Common"), ("sheep", "भेड़", "bhed", "मेंढी", "mendhi")),
    )
    
    for (animal_type, sub_type), keywords in animal_keywords:
        if any(keyword in text_lower for keyword in keywords):
            if animal_type in ANIMAL_SUBTYPES_MAP and sub_type in ANIMAL_SUBTYPES_MAP[animal_type]:
                print(f"✓ Fallback detected: {animal_type} - {sub_type}")
                return animal_type, sub_type
    
    print("❌ No animal detected")
    return None, None

# ---------------------------
# Symptom classification by type
# ---------------------------
def classify_symptoms_by_frequency(state):
    """Classify symptoms as unique, common, or intermediate based on frequency"""
    cache_key = f"symptom_freq_{len(state['candidate_records'])}"
    if cache_key in state:
        return state[cache_key]
    
    symptom_disease_map = {}
    
    for record in state["candidate_records"]:
        disease = record["Disease"]
        for symptom_norm in record['_normalized_symptoms']:
            if symptom_norm not in symptom_disease_map:
                symptom_disease_map[symptom_norm] = set()
            symptom_disease_map[symptom_norm].add(disease)
    
    total_diseases = len(set(r["Disease"] for r in state["candidate_records"]))
    
    unique_symptoms = []
    intermediate_symptoms = []
    common_symptoms = []
    
    for symptom_norm, diseases in symptom_disease_map.items():
        frequency = len(diseases) / total_diseases
        
        if len(diseases) <= 2:
            unique_symptoms.append(symptom_norm)
        elif frequency > 0.4:
            common_symptoms.append(symptom_norm)
        else:
            intermediate_symptoms.append(symptom_norm)
    
    result = {
        "unique": unique_symptoms,
        "intermediate": intermediate_symptoms,
        "common": common_symptoms,
        "symptom_disease_map": symptom_disease_map
    }
    
    state[cache_key] = result
    return result

# ---------------------------
# Smart symptom selection (ENHANCED for distinguishing diseases)
# ---------------------------
def select_next_symptom_smart(state):
    """
    Enhanced symptom selection that prioritizes UNIQUE/DISTINGUISHING symptoms
    to differentiate between diseases with overlapping symptoms
    """
    reported_norm = set(normalized_list(state["reported_symptoms"]))
    asked_norm = set(normalized_list(state["asked_symptoms"]))
    excluded = reported_norm | asked_norm
    
    symptom_info = classify_symptoms_by_frequency(state)
    symptom_scores = {}
    total = len(state["candidate_records"])
    
    if total == 0:
        return None
    
    disease_symptoms = [set(r['_normalized_symptoms']) for r in state["candidate_records"]]
    
    # Get all candidate symptoms
    candidate_symptoms = set()
    for symptom_set in disease_symptoms:
        candidate_symptoms.update(symptom_set - excluded)
    
    if not candidate_symptoms:
        return None
    
    # Calculate top diseases currently
    top_diseases = sorted(state["disease_probs"].items(), key=lambda x: x[1], reverse=True)
    top_2_diseases = [d for d, _ in top_diseases[:2]] if len(top_diseases) >= 2 else []
    
    # NEW: Identify distinguishing symptoms between top candidates
    distinguishing_symptoms = set()
    if len(top_2_diseases) == 2:
        disease1_record = next((r for r in state["candidate_records"] if r["Disease"] == top_2_diseases[0]), None)
        disease2_record = next((r for r in state["candidate_records"] if r["Disease"] == top_2_diseases[1]), None)
        
        if disease1_record and disease2_record:
            d1_symptoms = set(disease1_record['_normalized_symptoms'])
            d2_symptoms = set(disease2_record['_normalized_symptoms'])
            
            # Symptoms that are ONLY in disease1 or ONLY in disease2
            distinguishing_symptoms = (d1_symptoms - d2_symptoms) | (d2_symptoms - d1_symptoms)
            distinguishing_symptoms = distinguishing_symptoms - excluded
            
            if distinguishing_symptoms:
                print(f"\n🔍 Found {len(distinguishing_symptoms)} distinguishing symptoms between:")
                print(f"   {top_2_diseases[0]} vs {top_2_diseases[1]}")
    
    for symptom_norm in candidate_symptoms:
        has_count = sum(1 for symptom_set in disease_symptoms if symptom_norm in symptom_set)
        
        # Base information gain calculation
        information_gain = 1 - abs(has_count - total/2) / (total/2)
        
        # Base score from symptom frequency type
        if symptom_norm in symptom_info["unique"]:
            base_score = information_gain * 3.0
        elif symptom_norm in symptom_info["intermediate"]:
            base_score = information_gain * 2.0
        else:
            base_score = information_gain * 0.5
        
        # Boost if symptom can split candidates
        if 0 < has_count < total:
            base_score *= 1.2
        
        # 🔥 NEW: HUGE boost for distinguishing symptoms between top 2 candidates
        if symptom_norm in distinguishing_symptoms:
            base_score *= 5.0  # Massive priority boost
            print(f"   ⭐ Boosting '{symptom_norm}' (distinguishing symptom)")
        
        # 🔥 NEW: Penalize symptoms present in ALL top candidates (common symptoms)
        if has_count == total:
            base_score *= 0.1  # Heavy penalty for universal symptoms
        
        # 🔥 NEW: Boost symptoms that appear in fewer diseases (more specific)
        specificity_boost = 1.0 / (has_count + 1)
        base_score *= (1 + specificity_boost)
        
        symptom_scores[symptom_norm] = base_score
    
    if not symptom_scores:
        return None
    
    # Select best symptom
    best_symptom_norm = max(symptom_scores.items(), key=lambda x: x[1])[0]
    
    # Get original symptom text
    symptom = None
    if best_symptom_norm in symptom_to_original:
        symptom = symptom_to_original[best_symptom_norm]
    else:
        for record in state["candidate_records"]:
            for s in record["Symptoms"]:
                if normalize(s) == best_symptom_norm:
                    symptom = s
                    break
            if symptom:
                break
    
    if symptom:
        symptom_type = "unique" if best_symptom_norm in symptom_info["unique"] else \
                      "intermediate" if best_symptom_norm in symptom_info["intermediate"] else "common"
        is_distinguishing = best_symptom_norm in distinguishing_symptoms
        
        print(f"\n📋 Selected symptom: '{symptom}'")
        print(f"   Type: {symptom_type}")
        print(f"   Score: {symptom_scores[best_symptom_norm]:.2f}")
        print(f"   Distinguishing: {'YES ⭐' if is_distinguishing else 'NO'}")
        print(f"   Present in: {sum(1 for s in disease_symptoms if best_symptom_norm in s)}/{total} diseases\n")
        
        return symptom
    
    return None

# ---------------------------
# Symptom extraction (Enhanced for multilingual)
# ---------------------------
def extract_symptoms_with_llm(user_text, top_k=5):
    cache_key = f"symptoms_{hash(user_text)}_{top_k}"
    
    # Detect language for better prompting
    user_lang = detect_language(user_text)
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    
    prompt = f"""
You are a veterinary assistant. Extract ALL symptoms from this text and translate them to English.

Text (in {lang_map.get(user_lang, 'English')}): "{user_text}"

Common symptom translations:
- बुखार/ताप = fever
- खांसी/खोकला = cough  
- दस्त/अतिसार = diarrhea
- भूख नहीं/खात नाही = loss of appetite
- कमजोरी/अशक्तपणा = weakness
- सूजन/सूज = swelling
- दूध कम = reduced milk production
- नाक बहना = nasal discharge
- सांस लेने में दिक्कत = breathing difficulty

Return ONLY a JSON array of symptom strings in English, e.g. ["fever", "cough", "diarrhea"].
Extract maximum {top_k} most important symptoms.
"""
    
    response_text = cached_llm_call(cache_key, prompt)
    if response_text:
        try:
            match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if match:
                symptoms = json.loads(match.group(0))[:top_k]
                print(f"✓ LLM extracted symptoms: {symptoms}")
                return symptoms
        except Exception as e:
            print(f"⚠ LLM symptom extraction failed: {e}")
    return []

def extract_symptoms_semantic(farmer_text, top_k=5, threshold=0.6):
    if not farmer_text:
        return []
    farmer_emb = embedding_model.encode([farmer_text], convert_to_numpy=True)[0]
    sims = util.cos_sim(farmer_emb, all_symptom_embeddings)[0]
    matched = [all_symptoms[i] for i, sim in enumerate(sims) if sim >= threshold]
    return list(dict.fromkeys(matched))[:top_k]

def extract_symptoms_fuzzy(user_text, threshold=80):
    detected = set()
    words = re.findall(r'\w+', user_text.lower())
    for word in words:
        matches = process.extract(word, all_symptoms, scorer=fuzz.ratio, limit=3)
        for m in matches:
            if m[1] >= threshold:
                detected.add(m[0])
    return list(detected)

def extract_symptoms_multilingual(user_text):
    symptoms = extract_symptoms_with_llm(user_text)
    if symptoms:
        return symptoms
    detected = set()
    detected.update(extract_symptoms_semantic(user_text))
    detected.update(extract_symptoms_fuzzy(user_text))
    return list(detected)

# --- Separate Language Maps ---
EN_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15
}

HI_NUM = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10
}

MR_NUM = {
    "एक": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5,
    "सहा": 6, "सात": 7, "आठ": 8, "नऊ": 9, "दहा": 10
}

# --- Merge, but Hindi takes priority over Marathi where same word exists ---
NUM_WORDS_MAP = {**EN_NUM, **MR_NUM, **HI_NUM}

def replace_word_numbers(text: str) -> str:
    for word, num in NUM_WORDS_MAP.items():
        # Unicode-safe boundary: (?<!\w)word(?!\w)
        pattern = rf"(?<!\w){word}(?!\w)"
        text = re.sub(pattern, str(num), text, flags=re.IGNORECASE | re.UNICODE)
    return text

@lru_cache(maxsize=1000)
def extract_age_and_days_multilingual(user_text):
    processed_text = replace_word_numbers(user_text.lower())

    age_match = re.search(r'(\d+(\.\d+)?)\s*(year|yr|saal|साल|वर्ष)', processed_text, flags=re.UNICODE)
    days_match = re.search(r'(\d+)\s*(day|days|दिन|दिवस)', processed_text, flags=re.UNICODE)

    age = float(age_match.group(1)) if age_match else None
    illness_days = float(days_match.group(1)) if days_match else None

    return age, illness_days
# ---------------------------
# Generate counter question
# ---------------------------
def generate_counter_question_missing_info(missing_fields, lang):
    """Generate counter questions with COMPLETE translation"""
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    cache_key = f"missing_v2_{','.join(missing_fields)}_{lang}"
    
    prompt = f"""
You are a friendly veterinary assistant.
The farmer did not mention: {', '.join(missing_fields)}.

Generate SHORT, SIMPLE questions in {lang_map[lang]} to ask for this information.

CRITICAL REQUIREMENTS:
1. Questions must be COMPLETELY in {lang_map[lang]}
2. NO English words should appear in the questions
3. Use simple, farmer-friendly language
4. Make questions as SHORT as possible (one sentence each)

Field translations:
- "age" in Hindi = "उम्र" or "आयु"
- "age" in Marathi = "वय"
- "illness duration" in Hindi = "बीमारी की अवधि" or "कितने दिनों से बीमार"
- "illness duration" in Marathi = "आजाराचा कालावधी" or "किती दिवसांपासून आजारी"

Examples:
❌ BAD (English mixing): "What is the age of your animal?"
✅ GOOD (Hindi): "आपके पशु की उम्र क्या है?"
✅ GOOD (Marathi): "तुमच्या प्राण्याचे वय किती आहे?"

Return JSON with questions array in {lang_map[lang]}:
{{
  "questions": ["question 1 in {lang_map[lang]}", "question 2 in {lang_map[lang]}"]
}}

REMEMBER: Farmer speaks ONLY {lang_map[lang]}. NO English words.
"""
    
    response_text = cached_llm_call(cache_key, prompt, temperature=0.3)
    
    if response_text:
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                questions = data.get("questions", [])
                if questions:
                    # Validate no English mixing
                    if lang != "en":
                        for q in questions:
                            latin_words = len(re.findall(r'[a-zA-Z]{3,}', q))
                            if latin_words > 2:
                                print(f"⚠️ Detected English mixing in question: {q}")
                                break
                        else:
                            return questions
        except Exception as e:
            print(f"⚠️ LLM missing info generation failed: {e}")
    
    # Enhanced fallback with complete translation
    fallback = {
        "en": {
            "age": "What is the age of your animal (in years)?",
            "illness duration": "How many days has your animal been sick?"
        },
        "hi": {
            "age": "आपके पशु की उम्र कितनी है (वर्षों में)?",
            "illness duration": "आपका पशु कितने दिनों से बीमार है?"
        },
        "mr": {
            "age": "तुमच्या प्राण्याचे वय किती आहे (वर्षांमध्ये)?",
            "illness duration": "तुमचा प्राणी किती दिवसांपासून आजारी आहे?"
        }
    }
    
    questions = []
    for field in missing_fields:
        questions.append(fallback.get(lang, fallback["en"]).get(field, f"Please provide {field}"))
    return questions
# ---------------------------
# Probability update (Enhanced logging)
# ---------------------------
def update_probabilities(state, symptom, answer_yes):
    symptom_norm = normalize(symptom)
    
    print(f"\n{'='*50}")
    print(f"📊 Updating probabilities for symptom: '{symptom}'")
    print(f"   Answer: {'YES ✓' if answer_yes else 'NO ✗'}")
    print(f"{'='*50}")
    
    for record in state["candidate_records"]:
        disease = record["Disease"]
        old_prob = state["disease_probs"].get(disease, 1.0)
        prob = old_prob
        symptom_present = symptom_norm in record['_normalized_symptoms']
        
        if symptom_present:
            # Symptom is in disease record
            if answer_yes:
                prob *= 1.5  # Increase confidence
                change = "⬆️ INCREASED"
            else:
                prob *= 0.3  # Decrease confidence
                change = "⬇️ DECREASED"
        else:
            # Symptom is NOT in disease record
            if answer_yes:
                prob *= 0.3  # Decrease confidence (user reports symptom not in record)
                change = "⬇️ DECREASED (extra symptom)"
            else:
                change = "➡️ NO CHANGE"
        
        state["disease_probs"][disease] = prob
        
        if old_prob != prob:
            print(f"   {disease}: {old_prob:.3f} → {prob:.3f} {change}")
    
    # Normalize probabilities
    total = sum(state["disease_probs"].values())
    if total > 0:
        for d in state["disease_probs"]:
            state["disease_probs"][d] /= total
    
    # Show top 3 diseases after update
    top_3 = sorted(state["disease_probs"].items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n🏆 Top 3 diseases after update:")
    for i, (disease, prob) in enumerate(top_3, 1):
        print(f"   {i}. {disease}: {prob:.3f}")
    print(f"{'='*50}\n")

# ---------------------------
# Language detection
# ---------------------------
@lru_cache(maxsize=1000)
def detect_language(user_text):
    hindi_chars = bool(re.search(r'[\u0900-\u097F]', user_text))
    marathi_chars = bool(re.search(r'[\u0900-\u097F]', user_text))
    
    if hindi_chars or marathi_chars:
        marathi_words = ['आहे', 'आहेत', 'शेळी', 'म्हशी', 'मेंढी']
        if any(word in user_text for word in marathi_words):
            return "mr"
        return "hi"
    
    return "en"

# ---------------------------
# 🔥 FIXED: Generate question with COMPREHENSIVE observable advice
# ---------------------------
def generate_llm_question_with_advice(animal_type, sub_type, symptom, lang="en"):
    """
    Generate question with DETAILED observable signs that farmers can actually check
    WITHOUT medical equipment or veterinary knowledge
    COMPLETELY in the target language without mixing English words
    """
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    cache_key = f"question_v3_{animal_type}_{sub_type}_{symptom}_{lang}"
    
    prompt = f"""
You are a veterinary assistant helping a farmer who has NO medical knowledge or equipment.

Animal Type: {animal_type}
Animal Subtype: {sub_type}
Symptom to check: "{symptom}"
Target Language: {lang_map[lang]}

CRITICAL REQUIREMENTS:
1. Generate a SIMPLE yes/no question asking if symptom is PRESENT (YES = has symptom, NO = no symptom)
2. DO NOT use medical terms like "anemia", "respiratory distress", "septicemia" etc.
3. Ask about OBSERVABLE signs the farmer can SEE, HEAR, or FEEL
4. Provide DETAILED "How to Check" advice with SPECIFIC observable signs
5. **MOST IMPORTANT**: Generate the ENTIRE question and advice COMPLETELY in {lang_map[lang]}
6. **DO NOT keep any English words** like animal names or symptom names in the output
7. Translate EVERYTHING including animal type, subtype, and symptom names to {lang_map[lang]}

TRANSLATION GUIDELINES:
- Cattle = गाय-बैल/पशु (Hindi) = गुरे/पशू (Marathi)
- Cow = गाय (Hindi) = गाय (Marathi)
- Bull = बैल (Hindi) = बैल (Marathi)
- Buffalo = भैंस (Hindi) = म्हशी (Marathi)
- Goat = बकरी (Hindi) = शेळी (Marathi)
- Sheep = भेड़ (Hindi) = मेंढी (Marathi)
- Calf = बछड़ा (Hindi) = वासरू (Marathi)

Common symptom translations:
- fever = बुखार (Hindi) = ताप (Marathi)
- weight loss = वजन कम होना (Hindi) = वजन कमी होणे (Marathi)
- loss of appetite = भूख न लगना (Hindi) = भूक नसणे (Marathi)
- weakness = कमजोरी (Hindi) = अशक्तपणा (Marathi)
- diarrhea = दस्त (Hindi) = अतिसार (Marathi)
- cough = खांसी (Hindi) = खोकला (Marathi)
- swelling = सूजन (Hindi) = सूज (Marathi)
- breathing difficulty = सांस लेने में दिक्कत (Hindi) = श्वास घेण्यात अडचण (Marathi)
- nasal discharge = नाक से पानी आना (Hindi) = नाकातून स्राव (Marathi)
- reduced milk = दूध कम होना (Hindi) = दूध कमी होणे (Marathi)
- lameness = लंगड़ापन (Hindi) = लंगडेपणा (Marathi)

ADVICE MUST BE:
- SHORT and CONCISE (maximum 2-3 sentences)
- Focus on the MOST IMPORTANT observable signs only
- Tell what to check and what indicates YES vs NO

Examples of GOOD questions (FULLY translated):
❌ BAD (English mixing): "क्या आपके Cow में Weight loss है?"
✅ GOOD (Hindi): "क्या आपकी गाय का वजन कम हो रहा है?"
✅ GOOD (Marathi): "तुमच्या गायीचे वजन कमी होत आहे का?"

Examples of GOOD advice (SHORT and focused):
❌ BAD (too long): "Touch the ears, nose, and body - they should feel warm but not burning hot. Check if animal is shivering, standing with head down, or has dry hot nose. Reduced activity and eating less are also signs. Normal temperature feels warm but comfortable. Hot/burning feeling means fever."
✅ GOOD (concise): "कान और शरीर छूकर देखें - बहुत गर्म लगे तो हाँ। पशु कांप रहा हो या सुस्त हो तो हाँ।"

Return JSON in {lang_map[lang]} with COMPLETE translation:
{{
  "question": "Simple yes/no question about observable sign - FULLY in {lang_map[lang]}, NO English words",
  "advice": "SHORT advice (2-3 sentences max) - FULLY in {lang_map[lang]}, NO English words"
}}

REMEMBER: The farmer speaks ONLY {lang_map[lang]}. Every single word must be in {lang_map[lang]}.
"""
    
    response_text = cached_llm_call(cache_key, prompt, temperature=0.5)
    
    if response_text:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                question = parsed.get("question", "")
                advice = parsed.get("advice", "")
                
                # Validate that we got substantial advice
                if advice and len(advice) > 50:
                    # Extra validation: check if English words are still present (except JSON keys)
                    if lang != "en":
                        # Simple heuristic: check for Latin characters (indicating untranslated words)
                        latin_in_question = len(re.findall(r'[a-zA-Z]{3,}', question))
                        latin_in_advice = len(re.findall(r'[a-zA-Z]{3,}', advice))
                        
                        if latin_in_question > 2 or latin_in_advice > 5:
                            print(f"⚠️ Warning: Detected {latin_in_question} Latin words in question, {latin_in_advice} in advice. Regenerating...")
                            # Try once more with stronger prompt
                            response_text = cached_llm_call(
                                f"{cache_key}_retry", 
                                prompt + "\n\nREMINDER: ABSOLUTELY NO ENGLISH WORDS IN THE OUTPUT!", 
                                temperature=0.3
                            )
                            if response_text:
                                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                                if json_match:
                                    parsed = json.loads(json_match.group(0))
                                    question = parsed.get("question", question)
                                    advice = parsed.get("advice", advice)
                    
                    print(f"✅ Generated fully translated question ({len(question)} chars) and advice ({len(advice)} chars)")
                    return question, advice
                else:
                    print(f"⚠️ Advice too short ({len(advice)} chars), using fallback...")
        except Exception as e:
            print(f"⚠️ LLM question generation failed: {e}")
    
    # Enhanced fallback with complete translation
    print(f"🔄 Using fallback question generator for: {symptom}")
    question, advice = generate_fallback_question_with_advice(animal_type, sub_type, symptom, lang)
    return question, advice


def generate_fallback_question_with_advice(animal_type, sub_type, symptom, lang="en"):
    """
    Enhanced fallback with COMPLETE translation - NO English mixing
    Uses LLM to translate entire question and advice
    """
    # Use LLM to translate the complete question
    cache_key = f"fallback_translate_{animal_type}_{sub_type}_{symptom}_{lang}"
    
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    
    prompt = f"""
Translate this veterinary question to {lang_map[lang]} completely:

Animal: {sub_type} ({animal_type})
Symptom: {symptom}

Generate a simple yes/no question asking if the {sub_type} has this symptom.
Provide SHORT advice (2-3 sentences max) on how to check.

CRITICAL: Translate EVERYTHING to {lang_map[lang]}:
- Animal names must be in {lang_map[lang]}
- Symptom names must be in {lang_map[lang]}
- NO English words should remain
- Keep advice CONCISE and focused on key observable signs only

Common translations:
- Cattle/Cow = गाय (Hindi) = गाय (Marathi)
- Buffalo = भैंस (Hindi) = म्हशी (Marathi)
- Goat = बकरी (Hindi) = शेळी (Marathi)
- Bull = बैल (Hindi) = बैल (Marathi)
- Calf = बछड़ा (Hindi) = वासरू (Marathi)
- Sheep = भेड़ (Hindi) = मेंढी (Marathi)

Return JSON:
{{
  "question": "Fully translated question in {lang_map[lang]}",
  "advice": "SHORT advice in {lang_map[lang]} (2-3 sentences max)"
}}
"""
    
    response_text = cached_llm_call(cache_key, prompt, temperature=0.3)
    
    if response_text:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                question = parsed.get("question", "")
                advice = parsed.get("advice", "")
                
                if question and advice:
                    return question, advice
        except Exception as e:
            print(f"⚠️ Fallback translation failed: {e}")
    
    # Last resort: hardcoded templates (as final fallback only)
    symptom_lower = symptom.lower()
    
    # Comprehensive fallback templates with FULL translation and SHORT advice
    templates = {
        "en": {
            "default": (
                f"Does your {sub_type} have {symptom}?",
                f"Check your {sub_type} for signs of {symptom}. Answer YES if clearly present, NO if not."
            )
        },
        "hi": {
            "fever": (
                f"क्या आपकी {sub_type if sub_type == 'बकरी' or sub_type == 'गाय' or sub_type == 'भैंस' else 'पशु'} को बुखार है?",
                "कान और शरीर छूकर देखें - बहुत गर्म लगे तो हाँ। सुस्त हो तो हाँ।"
            ),
            "default": (
                f"क्या आपकी {sub_type if sub_type in ['बकरी', 'गाय', 'भैंस', 'बैल'] else 'पशु'} में यह लक्षण है?",
                "पशु को देखें। लक्षण साफ दिखे तो हाँ, न दिखे तो नहीं।"
            )
        },
        "mr": {
            "fever": (
                f"तुमच्या {sub_type if sub_type == 'शेळी' or sub_type == 'गाय' or sub_type == 'म्हशी' else 'प्राण्या'}ला ताप आहे का?",
                "कान आणि शरीर स्पर्श करा - खूप गरम वाटले तर होय। सुस्त असेल तर होय।"
            ),
            "default": (
                f"तुमच्या {sub_type if sub_type in ['शेळी', 'गाय', 'म्हशी', 'बैल'] else 'प्राण्या'}ला हे लक्षण आहे का?",
                "प्राण्याचे निरीक्षण करा। लक्षण स्पष्ट दिसले तर होय, नाहीतर नाही।"
            )
        }
    }
    
    lang_templates = templates.get(lang, templates["en"])
    
    # Try to find matching template
    for key in lang_templates:
        if key != "default" and key in symptom_lower:
            return lang_templates[key]
    
    return lang_templates["default"]

# Answer interpretation (FIXED for multilingual)
# ---------------------------
def interpret_answer_with_llm(answer_text, symptom, lang="en"):
    """Enhanced multilingual answer interpretation using LLM"""
    answer_norm = normalize(answer_text)
    
    # Extended multilingual indicators
    yes_indicators = {
        # English
        "yes", "yeah", "yep", "yup", "sure", "correct", "right", "true", "absolutely",
        "definitely", "ofcourse", "ok", "okay",
        # Hindi
        "हाँ", "हां", "हो", "ha", "haan", "han", "हा", "sahi", "सही", "theek", "ठीक",
        "bilkul", "बिल्कुल", "jarur", "जरूर", "accha", "अच्छा",
        # Marathi  
        "होय", "hoy", "ho", "आहे", "aahe", "ahe", "नक्की", "nakki", "avashya", "अवश्य"
    }
    
    no_indicators = {
        # English
        "no", "nope", "nah", "not", "never", "nahi", "neither", "negative",
        # Hindi
        "नहीं", "नही", "nahin", "nahi", "na", "नै", "nei", "mat", "मत", "galat", "गलत",
        # Marathi
        "नाही", "नको", "nako"
    }
    
    # First check direct indicators
    has_yes = any(ind in answer_norm for ind in yes_indicators)
    has_no = any(ind in answer_norm for ind in no_indicators)
    
    # If clear answer found, return it
    if has_yes and not has_no:
        print(f"✓ Direct YES detected: {answer_text}")
        return True, False
    elif has_no and not has_yes:
        print(f"✗ Direct NO detected: {answer_text}")
        return False, False
    
    # If ambiguous or no clear answer, use LLM
    cache_key = f"interpret_{hash(answer_text)}_{hash(symptom)}_{lang}"
    
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    
    prompt = f"""
You are analyzing a farmer's response to a veterinary question.

Question was about symptom: "{symptom}"
Farmer's answer: "{answer_text}"
Language: {lang_map.get(lang, "English")}

IMPORTANT: Determine if the farmer is saying YES (symptom is present) or NO (symptom is not present).

Common patterns:
- YES in English: yes, yeah, correct, right, true
- YES in Hindi: हाँ, हां, हो, सही, बिल्कुल
- YES in Marathi: होय, आहे, नक्की

- NO in English: no, not, never, nope
- NO in Hindi: नहीं, नही, गलत
- NO in Marathi: नाही, नको

Return ONLY JSON:
{{"answer": "yes"}} or {{"answer": "no"}}
"""
    
    response_text = cached_llm_call(cache_key, prompt)
    
    if response_text:
        try:
            # Clean response
            response_text = response_text.strip().lower()
            
            # Try JSON parsing
            match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                llm_answer = data.get("answer", "").lower()
                
                if "yes" in llm_answer or "हाँ" in llm_answer or "होय" in llm_answer:
                    print(f"✓ LLM detected YES: {answer_text}")
                    return True, False
                elif "no" in llm_answer or "नहीं" in llm_answer or "नाही" in llm_answer:
                    print(f"✗ LLM detected NO: {answer_text}")
                    return False, False
            
            # Fallback: check if response contains yes/no
            if "yes" in response_text or "\"yes\"" in response_text:
                print(f"✓ LLM fallback YES: {answer_text}")
                return True, False
            elif "no" in response_text or "\"no\"" in response_text:
                print(f"✗ LLM fallback NO: {answer_text}")
                return False, False
                
        except Exception as e:
            print(f"⚠ LLM interpretation error: {e}")
    
    # Final fallback: default to NO if nothing detected
    print(f"⚠ Ambiguous answer, defaulting to NO: {answer_text}")
    return False, False

# ---------------------------
# Hybrid prediction
# ---------------------------
def hybrid_predict(state):
    results = {}
    if not state["reported_symptoms"]:
        return {r["Disease"]: 0.0 for r in state["candidate_records"]}
    
    try:
        input_emb = np.mean(embedding_model.encode(state["reported_symptoms"], convert_to_numpy=True), axis=0)
        
        for record in state["candidate_records"]:
            avg_emb = np.mean(record["Symptom_Embeddings"], axis=0)
            sim = util.cos_sim(input_emb, avg_emb).item()
            X_input = np.hstack([avg_emb.reshape(1, -1), animal_encoder.transform([[record["Animal_Type"]]])])
            probs = clf.predict_proba(X_input)[0]
            disease_idx = disease_encoder.transform([record["Disease"]])[0]
            ml_prob = probs[disease_idx]
            weight = 0.5 + 0.3 * (sim - abs(ml_prob - 0.5))
            final_conf = weight * ml_prob + (1 - weight) * sim
            results[record["Disease"]] = max(results.get(record["Disease"], 0), final_conf)
        
        max_conf = max(results.values()) if results else 1
        for k in results:
            results[k] = round(float(results[k] / max_conf), 2)
    except Exception as e:
        print(f"Error in hybrid_predict: {e}")
        for record in state["candidate_records"]:
            results[record["Disease"]] = 0.5
    
    return results

# ---------------------------
# Generate disease report
# ---------------------------
def generate_disease_report_llm(animal_type, sub_type, lang, state, predictions, age, illness_days):
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    detected_disease = sorted_preds[0][0] if sorted_preds else "Unknown"
    probable_diseases = [d for d, _ in sorted_preds[1:4]]
    
    record = disease_to_record.get(detected_disease)
    is_critical = False
    if record:
        is_critical = record.get("Critical", False)
        if age is not None and illness_days is not None:
            if age < 0.5 or age > 7.5:
                if illness_days > record.get("Duration_days", 2):
                    is_critical = True
    
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    
    if is_critical:
        critical_advice_map = {
            "en": "⚠️ This disease seems critical. Please consult a veterinary doctor immediately.",
            "hi": "⚠️ यह बीमारी गंभीर लग रही है। कृपया तुरंत पशु चिकित्सक से परामर्श लें।",
            "mr": "⚠️ हा आजार गंभीर दिसतो आहे. कृपया त्वरित पशुवैद्यकीय डॉक्टरांचा सल्ला घ्या."
        }
    else:
        critical_advice_map = {
            "en": "This disease is not critical. Provide proper care at home. If there is no improvement, consult a doctor.",
            "hi": "यह बीमारी गंभीर नहीं है। घर पर उचित देखभाल प्रदान करें। यदि कोई सुधार नहीं होता है, तो डॉक्टर से परामर्श लें।",
            "mr": "हा आजार गंभीर नाही. घरी योग्य काळजी द्या. जर सुधारणा होत नसेल तर डॉक्टरांचा सल्ला घ्या."
        }
    
    critical_advice = critical_advice_map.get(lang, critical_advice_map["en"])
    cache_key = f"report_{detected_disease}_{animal_type}_{sub_type}_{lang}_{int(time.time())}"
    
    prompt = f"""
You are a veterinary expert.
Generate a detailed structured report in {lang_map[lang]}.

Animal: {sub_type} ({animal_type})
Symptoms: {', '.join(state['reported_symptoms'])}
Detected Disease: {detected_disease}
Age: {age} years
Sick for: {illness_days} days
Probable Diseases: {', '.join(probable_diseases)}

Return JSON:
{{
  "Animal_Type": "{animal_type}",
  "Sub_Type": "{sub_type}",
  "Age": {age},
  "Illness_Duration_Days": {illness_days},
  "Detected_Disease": "...",
  "Critical": true/false,
  "Probable_Diseases": [...],
  "Cause_of_Disease": "...",
  "Precautions": "...",
  "Care": "...",
  "Home_Remedies": "...",
  "Treatment_or_Medicine": "...",
  "Farmer_Advice": "..."
}}

Keep content simple, short, and farmer-friendly in {lang_map[lang]}.
"""
    
    response_text = cached_llm_call(cache_key, prompt)
    if response_text:
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                report = json.loads(match.group(0))
                report["Critical"] = is_critical
                report["Farmer_Advice"] = critical_advice
                report["Animal_Type"] = animal_type
                report["Sub_Type"] = sub_type
                report["Age"] = age
                report["Illness_Duration_Days"] = illness_days
                report["Reported_Symptoms"] = state['reported_symptoms']
                return report
        except Exception as e:
            print(f"Error parsing LLM report: {e}")
    
    fallback_report = {
        "Animal_Type": animal_type,
        "Sub_Type": sub_type,
        "Age": age,
        "Illness_Duration_Days": illness_days,
        "Detected_Disease": detected_disease,
        "Critical": is_critical,
        "Probable_Diseases": probable_diseases,
        "Cause_of_Disease": record.get("Cause_of_Disease", "") if record else "",
        "Precautions": record.get("Precautions", "") if record else "",
        "Care": record.get("Care", "") if record else "",
        "Home_Remedies": record.get("Home_Remedies", "") if record else "",
        "Treatment_or_Medicine": record.get("Treatment/Medicine", "") if record else "",
        "Farmer_Advice": critical_advice,
        "Reported_Symptoms": state['reported_symptoms']
    }
    return fallback_report

# ---------------------------
# Check if should stop asking questions (ENHANCED)
# ---------------------------
def should_stop_asking(state):
    """
    Enhanced stopping criteria:
    - Don't stop too early if top diseases are too close
    - Continue asking if there are distinguishing symptoms left
    """
    num_diseases = len(state["candidate_records"])
    min_questions = min(5, max(3, num_diseases // 2))
    
    # Don't stop before minimum questions
    if state["question_count"] < min_questions:
        return False
    
    # Hard limit: stop after 15 questions
    if state["question_count"] >= 15:
        print("⏹️ Stopping: Reached maximum question limit (15)")
        return True
    
    # Get top probabilities
    sorted_probs = sorted(state["disease_probs"].values(), reverse=True)
    max_prob = sorted_probs[0] if sorted_probs else 0
    
    # Stop if very confident (>95%) and asked minimum questions
    if max_prob >= 0.95 and state["question_count"] >= min_questions:
        print(f"⏹️ Stopping: High confidence ({max_prob:.2%})")
        return True
    
    # NEW: Check if top 2 diseases are too close
    if len(sorted_probs) >= 2:
        first = sorted_probs[0]
        second = sorted_probs[1]
        ratio = first / (second + 0.01)
        
        # If top disease is 3x more likely than second, and we've asked enough
        if ratio > 3.0 and state["question_count"] >= min_questions:
            print(f"⏹️ Stopping: Clear winner (ratio: {ratio:.1f}x)")
            return True
        
        # NEW: If top 2 are very close, keep asking (unless we've asked many questions)
        if ratio < 1.5 and state["question_count"] < 12:
            print(f"🔄 Continuing: Top 2 diseases too close (ratio: {ratio:.1f}x)")
            return False
    
    # NEW: Check if there are distinguishing symptoms left to ask
    top_diseases = sorted(state["disease_probs"].items(), key=lambda x: x[1], reverse=True)[:2]
    if len(top_diseases) == 2:
        disease1_record = next((r for r in state["candidate_records"] if r["Disease"] == top_diseases[0][0]), None)
        disease2_record = next((r for r in state["candidate_records"] if r["Disease"] == top_diseases[1][0]), None)
        
        if disease1_record and disease2_record:
            reported_norm = set(normalized_list(state["reported_symptoms"]))
            asked_norm = set(normalized_list(state["asked_symptoms"]))
            excluded = reported_norm | asked_norm
            
            d1_symptoms = set(disease1_record['_normalized_symptoms'])
            d2_symptoms = set(disease2_record['_normalized_symptoms'])
            
            # Distinguishing symptoms not yet asked
            distinguishing = ((d1_symptoms - d2_symptoms) | (d2_symptoms - d1_symptoms)) - excluded
            
            if distinguishing and state["question_count"] < 12:
                print(f"🔄 Continuing: {len(distinguishing)} distinguishing symptoms remain")
                return False
    
    # Default: stop if we've asked 8+ questions and have some confidence
    if state["question_count"] >= 8 and max_prob > 0.5:
        print(f"⏹️ Stopping: Sufficient questions ({state['question_count']}) with reasonable confidence")
        return True
    
    return False

# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.route('/api/start', methods=['POST'])
def start_conversation():
    """Initialize conversation with enhanced animal detection"""
    cleanup_old_sessions()
    
    try:
        data = request.json
        user_text = data.get('text', '').strip()
        
        if not user_text:
            return jsonify({"error": "No text provided"}), 400
        
        user_lang = detect_language(user_text)
        print(f"Detected language: {user_lang}")
        
        animal_type, sub_type = extract_animal_type_with_llm(user_text)
        
        if not animal_type or not sub_type:
            error_messages = {
                "en": "Could not detect animal type. Please mention the animal (e.g., cow, buffalo, goat) clearly.",
                "hi": "पशु का प्रकार पहचाना नहीं जा सका। कृपया पशु (जैसे गाय, भैंस, बकरी) का स्पष्ट उल्लेख करें।",
                "mr": "प्राण्याचा प्रकार ओळखता आला नाही. कृपया प्राणी (उदा. गाय, म्हशी, बकरी) स्पष्टपणे नमूद करा."
            }
            return jsonify({
                "error": error_messages.get(user_lang, error_messages["en"]),
                "retry": True,
                "available_animals": list(ANIMAL_SUBTYPES_MAP.keys())
            }), 400
        
        animal_key = (normalize(animal_type), normalize(sub_type))
        candidate_records = animal_to_records.get(animal_key, [])
        
        if not candidate_records:
            available_subtypes_msg = {
                "en": f"No data available for {animal_type} - {sub_type}. Available types: {', '.join(ANIMAL_SUBTYPES_MAP.get(animal_type, []))}",
                "hi": f"{animal_type} - {sub_type} के लिए कोई डेटा उपलब्ध नहीं है। उपलब्ध प्रकार: {', '.join(ANIMAL_SUBTYPES_MAP.get(animal_type, []))}",
                "mr": f"{animal_type} - {sub_type} साठी डेटा उपलब्ध नाही. उपलब्ध प्रकार: {', '.join(ANIMAL_SUBTYPES_MAP.get(animal_type, []))}"
            }
            return jsonify({
                "error": available_subtypes_msg.get(user_lang, available_subtypes_msg["en"]),
                "detected_animal": animal_type,
                "detected_subtype": sub_type,
                "available_subtypes": ANIMAL_SUBTYPES_MAP.get(animal_type, []),
                "retry": True
            }), 400
        
        reported_symptoms = extract_symptoms_multilingual(user_text)
        if not reported_symptoms:
            symptom_prompts = {
                "en": "Please describe the symptoms more clearly (e.g., fever, not eating, coughing).",
                "hi": "कृपया लक्षणों को अधिक स्पष्ट रूप से बताएं (जैसे बुखार, खाना नहीं खा रहा, खांसी)।",
                "mr": "कृपया लक्षणे अधिक स्पष्टपणे सांगा (उदा. ताप, खात नाही, खोकला)."
            }
            return jsonify({
                "error": "Could not detect symptoms",
                "question": symptom_prompts.get(user_lang, symptom_prompts["en"]),
                "animal_detected": animal_type,
                "subtype_detected": sub_type,
                "retry": True
            }), 400
        
        age, illness_days = extract_age_and_days_multilingual(user_text)
        
        session_id = create_session_id()
        state = {
            "animal_type": animal_type,
            "sub_type": sub_type,
            "reported_symptoms": reported_symptoms.copy(),
            "asked_symptoms": [],
            "candidate_records": candidate_records,
            "disease_probs": {r["Disease"]: 1.0 for r in candidate_records},
            "language": user_lang,
            "age": age,
            "illness_days": illness_days,
            "question_count": 0,
            "missing_info_questions": [],
            "session_id": session_id
        }
        set_session(session_id, state)
        
        print(f"✓ Session created: {session_id}")
        print(f"  Animal: {animal_type} - {sub_type}")
        print(f"  Initial symptoms: {reported_symptoms}")
        print(f"  Candidate diseases: {len(candidate_records)}")
        
        response_data = {
            "session_id": session_id,
            "animal_type": animal_type,
            "sub_type": sub_type,
            "detected_symptoms": reported_symptoms.copy(),
            "age": age,
            "illness_days": illness_days
        }
        
        missing_fields = []
        if age is None:
            missing_fields.append("age")
        if illness_days is None:
            missing_fields.append("illness duration")
        
        if missing_fields:
            questions = generate_counter_question_missing_info(missing_fields, user_lang)
            state["missing_info_questions"] = questions
            state["missing_fields"] = missing_fields
            set_session(session_id, state)
            response_data["missing_info"] = True
            response_data["missing_fields"] = missing_fields
            response_data["question"] = questions[0] if questions else "What is the age of your animal?"
            response_data["question_type"] = "missing_info"
            return jsonify(response_data)
        
        next_symptom = select_next_symptom_smart(state)
        if next_symptom:
            question, advice = generate_llm_question_with_advice(animal_type, sub_type, next_symptom, lang=user_lang)
            response_data["question"] = question
            response_data["advice"] = advice
            response_data["symptom"] = next_symptom
            response_data["question_type"] = "symptom"
            return jsonify(response_data)
        
        return jsonify({**response_data, "ready_for_report": True})
    
    except Exception as e:
        print(f"Error in start_conversation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/answer', methods=['POST'])
def answer_question():
    """Handle answer to follow-up question"""
    try:
        data = request.json
        session_id = data.get('session_id')
        answer = data.get('answer', '').strip()
        question_type = data.get('question_type', 'symptom')
        
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400
        
        state = get_session(session_id)
        if not state:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        if question_type == 'missing_info':
            missing_fields = state.get('missing_fields', []).copy()
            
            if 'age' in missing_fields:
                try:
                    age_val = float(re.findall(r'\d+(?:\.\d+)?', answer)[0])
                    state['age'] = age_val
                    missing_fields.remove('age')
                except:
                    state['age'] = age_summary.get(state['animal_type'], {}).get("avg_age", 3.0)
                    if 'age' in missing_fields:
                        missing_fields.remove('age')
            
            elif 'illness duration' in missing_fields:
                try:
                    days_val = float(re.findall(r'\d+', answer)[0])
                    state['illness_days'] = days_val
                    missing_fields.remove('illness duration')
                except:
                    state['illness_days'] = 2.0
                    if 'illness duration' in missing_fields:
                        missing_fields.remove('illness duration')
            
            state['missing_fields'] = missing_fields
            set_session(session_id, state)
            
            if missing_fields:
                questions = generate_counter_question_missing_info(missing_fields, state['language'])
                return jsonify({
                    "missing_info": True,
                    "missing_fields": missing_fields,
                    "question": questions[0] if questions else "Please provide the information.",
                    "question_type": "missing_info"
                })
            
            next_symptom = select_next_symptom_smart(state)
            if next_symptom:
                question, advice = generate_llm_question_with_advice(
                    state['animal_type'], state['sub_type'], next_symptom, lang=state['language']
                )
                return jsonify({
                    "question": question,
                    "advice": advice,
                    "symptom": next_symptom,
                    "question_type": "symptom"
                })
            
            return jsonify({"ready_for_report": True})
        
        # Handle symptom questions
        symptom = data.get('symptom')
        if not symptom:
            return jsonify({"error": "No symptom provided"}), 400
        
        # Pass language to interpretation function
        answer_yes, _ = interpret_answer_with_llm(answer, symptom, lang=state['language'])
        state["asked_symptoms"].append(symptom)
        if answer_yes:
            if normalize(symptom) not in normalized_list(state["reported_symptoms"]):
                state["reported_symptoms"].append(symptom)
                print(f"✓ Symptom confirmed: {symptom}")
            else:
                print(f"⚠ Symptom already reported: {symptom}")
        else:
            print(f"✗ Symptom absent: {symptom}")
        
        update_probabilities(state, symptom, answer_yes)
        state["question_count"] += 1
        set_session(session_id, state)
        
        top_diseases = sorted(state["disease_probs"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"📊 After Q{state['question_count']}: Top diseases: {[(d, f'{p:.2f}') for d, p in top_diseases]}")
        
        if should_stop_asking(state):
            print(f"✓ Stopping after {state['question_count']} questions")
            return jsonify({"ready_for_report": True})
        
        next_symptom = select_next_symptom_smart(state)
        if not next_symptom:
            print("✓ No more relevant symptoms to ask")
            return jsonify({"ready_for_report": True})
        
        question, advice = generate_llm_question_with_advice(
            state['animal_type'], state['sub_type'], next_symptom, lang=state['language']
        )
        
        return jsonify({
            "question": question,
            "advice": advice,
            "symptom": next_symptom,
            "question_type": "symptom"
        })
    
    except Exception as e:
        print(f"Error in answer_question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate final disease report"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400
        
        state = get_session(session_id)
        if not state:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        # Generate predictions
        predictions = hybrid_predict(state)
        
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        top_predictions = []
        for disease, conf in sorted_preds:
            record = disease_to_record.get(disease)
            critical = record.get("Critical", False) if record else False
            top_predictions.append({
                "disease": disease,
                "confidence": conf,
                "critical": critical
            })
        
        print(f"📋 Final Report - Top Predictions:")
        for pred in top_predictions[:3]:
            print(f"  {pred['disease']}: {pred['confidence']:.2f} {'⚠️' if pred['critical'] else ''}")
        sys.stdout.flush()
        
        # Generate detailed report
        report = generate_disease_report_llm(
            state['animal_type'],
            state['sub_type'],
            state['language'],
            state,
            predictions,
            state.get('age'),
            state.get('illness_days')
        )
        
        if not report:
            print("⚠️ generate_disease_report_llm() returned None or empty dict")
            sys.stdout.flush()
            report = {}
        
        # Ensure all fields are present
        report['Reported_Symptoms'] = state['reported_symptoms'].copy()
        report['Animal_Type'] = state['animal_type']
        report['Sub_Type'] = state['sub_type']
        report['Age'] = state.get('age')
        report['Illness_Duration_Days'] = state.get('illness_days')
        
        print("✅ Final Report to return:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.stdout.flush()
        
        # Delete session after generating report to prevent data leakage
        delete_session(session_id)
        
        return jsonify(report)
    
    except Exception as e:
        print(f"Error in generate_report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    cleanup_old_sessions()
    return jsonify({
        "status": "healthy",
        "sessions": len(sessions),
        "available_animals": ANIMAL_SUBTYPES_MAP,
        "cache_size": len(llm_cache),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/animals', methods=['GET'])
def get_available_animals():
    """Get list of all available animal types and subtypes"""
    return jsonify({
        "animal_types": ANIMAL_SUBTYPES_MAP,
        "total_records": len(records)
    })

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Manually clear all caches"""
    clear_llm_cache()
    normalize.cache_clear()
    extract_age_and_days_multilingual.cache_clear()
    detect_language.cache_clear()
    cleanup_old_sessions()
    return jsonify({
        "status": "success",
        "message": "All caches cleared"
    })

@app.route('/api/end_session', methods=['POST'])
def end_session():
    """Manually end a session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400
        
        delete_session(session_id)
        return jsonify({"status": "success", "message": "Session ended"})
    
    except Exception as e:
        print(f"Error in end_session: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def home():
    return """
    <h1>🐄 Veterinary Diagnosis System (Powered by Google Gemini)</h1>
    <p>API is running. Try these endpoints:</p>
    <ul>
        <li><a href="/api/health">/api/health</a> - Health check</li>
        <li><a href="/api/animals">/api/animals</a> - Available animals</li>
        <li>/api/start - Start diagnosis (POST)</li>
        <li>/api/answer - Answer questions (POST)</li>
        <li>/api/report - Generate report (POST)</li>
        <li>/api/clear_cache - Clear all caches (POST)</li>
        <li>/api/end_session - End session (POST)</li>
    </ul>
    """

# ---------------------------
# Initialize optimizations on startup
# ---------------------------
print("\n" + "="*60)
print("🐄 Veterinary Diagnosis System Starting (OPTIMIZED & FIXED)...")
print("🚀 Now powered by Google Gemini API!")
print("="*60)
print("\n🔧 Running optimization routines...")

precompute_symptom_mappings()
precompute_record_structures()
create_lookup_indexes()

print(f"\n📊 Loaded {len(records)} medical records")
print(f"🐾 Available Animal Types: {len(ANIMAL_SUBTYPES_MAP)}")
for animal_type, subtypes in ANIMAL_SUBTYPES_MAP.items():
    print(f"  • {animal_type}: {len(subtypes)} subtypes")

print("\n" + "="*60)
print("🚀 Server starting on http://0.0.0.0:5001")
print("="*60 + "\n")

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
