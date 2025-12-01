import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import os
from datetime import datetime
import base64
from io import BytesIO

# -------------------- CONFIG --------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cnn_symptoms_model.keras")
model = tf.keras.models.load_model(model_path)

api_key = "AIzaSyABMWswAPxrNsHsLrekuyqGHeXKJyY7Hoc"
genai.configure(api_key=api_key)
llm = genai.GenerativeModel("gemini-2.5-flash")

# -------------------- LANGUAGE MAPPING --------------------
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "mr": "Marathi (मराठी)"
}

LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in English.",
    "hi": "Respond in Hindi (हिंदी). Use Devanagari script for Hindi words. Include English technical terms in parentheses when needed.",
    "mr": "Respond in Marathi (मराठी). Use Devanagari script for Marathi words. Include English technical terms in parentheses when needed."
}

# -------------------- DISEASE DICTIONARY --------------------
goat_skin_diseases = {
    "Ringworm": {
        "symptoms": ["Circular bald patches", "Crusty skin", "Itching", "Hair breakage", "Scaling"],
        "severity_progression": {
            "0-14 days": "Mild",
            "15-30 days": "Moderate",
            "31+ days": "Critical"
        },
        "cause": "Fungal infection (Trichophyton)",
        "zoonotic": "Yes",
        "precautions": "Isolate infected animals",
        "care": "Maintain dry, clean skin",
        "home": "Neem leaf paste",
        "treatment": "Topical antifungals such as clotrimazole or miconazole; clean lesions with antiseptic.",
        "body_parts": "Head, neck, face, ears, around eyes, limbs"
    },
    "Fly Strike": {
        "symptoms": ["Eggs in wounds", "Skin decay", "Pain", "Fly attraction", "Sores"],
        "severity_progression": {
            "0-3 days": "Mild",
            "4-7 days": "Moderate",
            "8+ days": "Critical"
        },
        "cause": "Blowflies",
        "zoonotic": "No",
        "precautions": "Keep wound clean",
        "care": "Apply repellents",
        "home": "Camphor oil",
        "treatment": "Remove larvae, clean wound, apply topical insecticide or fly repellent; systemic antibiotics if secondary infection.",
        "body_parts": "Perineal region, tail, wounds, back, hooves"
    },
    "Lumpy Skin Disease": {
        "symptoms": ["Firm skin nodules", "Fever", "Enlarged lymph nodes", "Crusts", "Swelling"],
        "severity_progression": {
            "0-7 days": "Mild",
            "8-14 days": "Moderate",
            "15+ days": "Critical"
        },
        "cause": "LSDV (capripoxvirus) – rare in goats",
        "zoonotic": "No",
        "precautions": "Avoid contact with cattle",
        "care": "Isolate affected goats",
        "home": "Neem paste",
        "treatment": "Supportive care, anti-inflammatories, antibiotics for secondary bacterial infections; consult vet for specific antiviral/supportive therapy.",
        "body_parts": "Neck, back, limbs, udder, face"
    }
}

num_symptoms_features = 20
class_names = ['Fly Strike', 'HealthyGoatAug', 'InvalidGoatAug', 'Lumpy Skin Disease', 'Ringworm']

# Session storage (in production, use Redis or database)
sessions = {}

# -------------------- HELPER FUNCTIONS --------------------

def is_valid_image(img):
    """Check if uploaded image is valid and not irrelevant."""
    try:
        if img.size[0] < 50 or img.size[1] < 50:
            raise ValueError("⚠ Image too small. Upload a clear goat skin image.")
        img_array = np.array(img.convert("L"))
        if img_array.std() < 1:
            raise ValueError("⚠ Image appears blank or uniform. Upload a clearer image.")
        return True
    except Exception as e:
        raise ValueError(f"⚠ Invalid or unreadable image: {e}")

def get_severity_for_duration(duration_days, severity_dict):
    for day_range, severity in severity_dict.items():
        if "+" in day_range:
            start = int(day_range.split("+")[0])
            if duration_days >= start:
                return severity
        else:
            first_part = day_range.split(" ")[0]
            start_end = first_part.split("-")
            if len(start_end) == 2:
                start, end = map(int, start_end)
                if start <= duration_days <= end:
                    return severity
    return "Unknown"

def predict_top3(img):
    if not is_valid_image(img):
        raise ValueError("⚠ Uploaded image is invalid or unrelated.")
    
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    symptoms_input_array = np.zeros((1, num_symptoms_features))
    preds = model.predict([img_array, symptoms_input_array], verbose=0)[0]
    
    top3_indices = preds.argsort()[-3:][::-1]
    top3 = [{"disease": class_names[i], "confidence": float(preds[i])} for i in top3_indices]
    return top3

def ask_llm_for_counter_questions(user_description, top3_diseases, language="en"):
    """Generate clarifying counter questions in the specified language."""
    data = {d["disease"]: goat_skin_diseases.get(d["disease"], {}) for d in top3_diseases}
    
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])

    prompt = f"""
    You are a veterinary assistant.
    A goat owner provided this description of symptoms:
    "{user_description}"

    The model predicts the top 3 possible diseases as:
    {json.dumps(top3_diseases, indent=2)}

    Refer to this goat disease database:
    {json.dumps(data, indent=2)}

    {lang_instruction}
    
    Generate 3–5 concise, specific follow-up questions that would help identify the correct disease.
    Each question should be based on the description and differences among predicted diseases.
    For technical medical terms, include the English word in parentheses.
    Avoid generic questions like "Is the goat sick?" — ask about severity, behavior, appearance, or body parts.
    
    IMPORTANT: 
    - Output ONLY the numbered questions, nothing else.
    - Do NOT include any introductory text, explanations, or conclusions.
    - Do NOT say things like "Here are some questions" or "To help diagnose".
    - Start directly with "1. [first question]"
    
    Format:
    1. [Question 1]
    2. [Question 2]
    3. [Question 3]
    """
    response = llm.generate_content(prompt)
    text = response.text.strip()
    
    # Remove any introductory lines before the questions
    lines = text.split('\n')
    question_lines = []
    found_first_question = False
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+[\.\)]\s+', line):
            found_first_question = True
            question_lines.append(line)
        elif found_first_question and line:
            question_lines.append(line)
    
    full_text = '\n'.join(question_lines)
    questions = [q.strip() for q in re.split(r'\n\d+[\.\)]\s*', full_text) if q.strip()]
    
    return questions

def extract_symptoms_from_description(description, answers):
    """Extract meaningful symptom phrases from user description and answers."""
    combined_text = ""

    if description:
        combined_text += description.strip() + " "
    if answers:
        combined_text += " ".join(str(v).strip() for v in answers.values())

    combined_text = combined_text.lower()

    symptom_keywords = [
        "fever", "patch", "bald", "crust", "itch", "scratch", "pain", "sore",
        "swelling", "nodule", "lump", "lesion", "hair loss", "scaling", "wound",
        "bleeding", "redness", "weakness", "discharge", "dry skin", "flaky",
        "ulcer", "scab", "pustule", "boil", "swollen", "limping", "loss of appetite",
        # Hindi keywords
        "बुखार", "खुजली", "घाव", "सूजन", "दर्द",
        # Marathi keywords
        "ताप", "खाज", "जखम", "सूज", "वेदना"
    ]

    detected_symptoms = []
    sentences = re.split(r'[.,;!?]', combined_text)
    for s in sentences:
        s = s.strip()
        if any(k in s for k in symptom_keywords):
            detected_symptoms.append(s.capitalize())

    unique_symptoms = []
    for s in detected_symptoms:
        if s not in unique_symptoms and len(s.split()) <= 10:
            unique_symptoms.append(s)

    if not unique_symptoms:
        if description:
            unique_symptoms = [description.strip()]
        elif answers:
            unique_symptoms = list(answers.values())[:3]
        else:
            unique_symptoms = ["Symptoms not clearly described"]

    return unique_symptoms

def parse_llm_diagnosis(text):
    """Parse structured diagnosis from LLM response with improved section detection."""
    sections = {
        "cause": "",
        "precautions": "",
        "care": "",
        "home_remedies": "",
        "treatment": "",
        "symptoms": []
    }
    
    # Split by common section headers (case insensitive)
    text_upper = text.upper()
    
    # Find section positions
    section_markers = {
        'cause': ['CAUSE:', 'CAUSE OF DISEASE:', 'कारण:', 'कारण'],
        'precautions': ['PRECAUTION', 'PRECAUTION:', 'सावधानी', 'खबरदारी'],
        'care': ['CARE:', 'SUPPORTIVE CARE:', 'देखभाल:', 'देखभाल'],
        'home_remedies': ['HOME', 'HOME_REMED', 'घरेलू', 'घरगुती'],
        'treatment': ['TREATMENT:', 'MEDICINE:', 'उपचार:', 'औषध:'],
        'symptoms': ['SYMPTOM', 'लक्षण']
    }
    
    # Find all section positions
    section_positions = []
    for section_key, markers in section_markers.items():
        for marker in markers:
            pos = text_upper.find(marker)
            if pos != -1:
                section_positions.append((pos, section_key, marker))
                break  # Found this section, move to next
    
    # Sort by position
    section_positions.sort(key=lambda x: x[0])
    
    # Extract content between sections
    for i, (pos, section_key, marker) in enumerate(section_positions):
        start_pos = pos + len(marker)
        
        # Find end position (start of next section or end of text)
        if i + 1 < len(section_positions):
            end_pos = section_positions[i + 1][0]
        else:
            end_pos = len(text)
        
        # Extract content
        content = text[start_pos:end_pos].strip()
        
        # Clean up content
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and section headers
            if line and not any(m in line.upper() for markers in section_markers.values() for m in markers):
                cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines).strip()
        
        # Store in appropriate section
        if section_key == 'symptoms':
            # Extract bullet points for symptoms
            for line in cleaned_lines:
                if line.startswith('•') or line.startswith('-') or line.startswith('*') or line.startswith('–'):
                    symptom = line.lstrip('•-*–').strip()
                    if symptom:
                        sections['symptoms'].append(symptom)
                elif re.match(r'^\d+\.', line):
                    symptom = re.sub(r'^\d+\.\s*', '', line).strip()
                    if symptom:
                        sections['symptoms'].append(symptom)
        else:
            sections[section_key] = content
    
    # Debug print
    print(f"DEBUG - Parsed sections: {list(sections.keys())}")
    print(f"DEBUG - Care section length: {len(sections.get('care', ''))}")
    print(f"DEBUG - Care content: {sections.get('care', 'EMPTY')[:200]}")
    
    return sections

def get_treatment_text(final_disease, parsed_treatment_text, language="en"):
    """Return treatment text in the specified language."""
    if parsed_treatment_text:
        t = parsed_treatment_text.strip()
        if len(t) > 10 and 'not specified' not in t.lower():
            return t

    dict_treatment = goat_skin_diseases.get(final_disease, {}).get("treatment", "")
    if dict_treatment and len(dict_treatment) > 5:
        try:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])
            
            prompt = f"""
            You are a licensed veterinary practitioner. The disease is: {final_disease}.
            Known short treatment note: "{dict_treatment}"

            {lang_instruction}

            Produce a concise, actionable "TREATMENT" paragraph (2-4 sentences) that:
            - Mentions the main medicines (generic names only) implied by the short note.
            - Gives simple dosage guidance when standard (e.g., "apply topical X twice daily", "inject Y at Z mg/kg once"), but do NOT invent exact mg/kg doses for drugs not commonly dosed without vet consultation. If uncertain, say "consult local veterinarian for dosage".
            - Include one sentence about when to seek veterinary care (red flags).
            - Keep it concise (max ~60-90 words).
            - Include English technical terms in parentheses if not writing in English.
            Return ONLY the treatment paragraph, no headings.
            """
            resp = llm.generate_content(prompt)
            text = resp.text.strip()
            if text:
                return text
        except Exception as e:
            return dict_treatment

    if dict_treatment:
        return dict_treatment

    return "Treatment details not available; consult a veterinarian."

def get_care_text(final_disease, parsed_care_text, language="en"):
    """Return care text in the specified language with fallback."""
    if parsed_care_text:
        c = parsed_care_text.strip()
        if len(c) > 10 and 'not specified' not in c.lower():
            return c

    # Fallback to dictionary
    dict_care = goat_skin_diseases.get(final_disease, {}).get("care", "")
    if dict_care and len(dict_care) > 5:
        try:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])
            
            prompt = f"""
            You are a licensed veterinary practitioner. The disease is: {final_disease}.
            Known short care note: "{dict_care}"

            {lang_instruction}

            Expand this into a detailed "CARE" paragraph (3-5 sentences) that includes:
            - Specific housing and environment recommendations
            - Temperature monitoring guidelines (mention normal range: 101.5-103.5°F / 38.6-39.7°C)
            - Nutrition and hydration advice
            - Daily monitoring checklist (appetite, posture, skin condition)
            - Isolation or hygiene practices if relevant
            - Include English technical terms in parentheses if not writing in English.
            
            Keep it practical and actionable. Return ONLY the care paragraph, no headings.
            """
            resp = llm.generate_content(prompt)
            text = resp.text.strip()
            if text and len(text) > 20:
                return text
        except Exception as e:
            print(f"ERROR in get_care_text LLM call: {e}")

    if dict_care:
        return dict_care

    # Ultimate fallback
    default_care = {
        "en": "Provide clean, dry housing with good ventilation. Monitor temperature daily (normal: 101.5-103.5°F). Ensure fresh water and quality feed are always available. Check for changes in appetite, behavior, or skin condition. Maintain hygiene in living area.",
        "hi": "स्वच्छ, सूखा आवास और अच्छे वेंटिलेशन के साथ प्रदान करें। दैनिक तापमान की निगरानी करें (सामान्य: 101.5-103.5°F)। ताजा पानी और गुणवत्तापूर्ण चारा हमेशा उपलब्ध रखें। भूख, व्यवहार या त्वचा की स्थिति में बदलाव की जांच करें। रहने वाले क्षेत्र में स्वच्छता बनाए रखें।",
        "mr": "स्वच्छ, कोरड्या घरासह चांगले वायुवीजन प्रदान करा. दररोज तापमान तपासा (सामान्य: 101.5-103.5°F). ताजे पाणी आणि दर्जेदार खाद्य नेहमी उपलब्ध ठेवा. भूक, वर्तन किंवा त्वचा स्थितीतील बदल तपासा. राहण्याच्या जागेत स्वच्छता राखा."
    }
    
    return default_care.get(language, default_care["en"])

# -------------------- API ENDPOINTS --------------------

@app.route('/api/start', methods=['POST'])
def start_session():
    """Initialize a new diagnosis session with image upload."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # Get language from request (default to English)
        language = request.form.get('language', 'en')
        if language not in LANGUAGE_NAMES:
            language = 'en'
        
        file = request.files['image']
        img = Image.open(file.stream)
        
        # Predict top 3 diseases
        top3 = predict_top3(img)
        
        if top3[0]["disease"] == "InvalidGoatAug":
            return jsonify({
                "error": "Upload affected part image ⚠",
                "message": "Please upload a clear image of the affected area."
            }), 400
        
        # Generate session ID
        session_id = str(datetime.now().timestamp())
        
        # Store session data with language
        sessions[session_id] = {
            "top3": top3,
            "counter_questions": [],
            "user_answers": {},
            "current_question_index": -1,
            "user_description": None,
            "duration_days": None,
            "age": None,
            "stage": "initial",
            "language": language
        }
        
        print(f"DEBUG - New session created: {session_id}, Language: {language}")
        
        # Language-specific initial messages
        initial_messages = {
            "en": {
                "message": "Image analyzed successfully! Please provide information about the goat.",
                "question": "Please describe the goat's symptoms in detail:"
            },
            "hi": {
                "message": "छवि का विश्लेषण सफलतापूर्वक किया गया! कृपया बकरी के बारे में जानकारी प्रदान करें।",
                "question": "कृपया बकरी के लक्षणों का विस्तार से वर्णन करें:"
            },
            "mr": {
                "message": "प्रतिमा यशस्वीरित्या विश्लेषित केली! कृपया शेळीबद्दल माहिती द्या.",
                "question": "कृपया शेळीच्या लक्षणांचे तपशीलवार वर्णन करा:"
            }
        }
        
        msg = initial_messages.get(language, initial_messages["en"])
        
        return jsonify({
            "session_id": session_id,
            "message": msg["message"],
            "question": msg["question"],
            "stage": "initial",
            "language": language
        })
    
    except Exception as e:
        print(f"ERROR in start_session: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/answer', methods=['POST'])
def process_answer():
    """Process user answers and return next question or final diagnosis."""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        session_id = data.get('session_id')
        answer = data.get('answer')
        
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
            
        if not answer:
            return jsonify({"error": "Missing answer"}), 400
        
        if session_id not in sessions:
            return jsonify({"error": "Invalid or expired session. Please start a new diagnosis."}), 400
        
        session = sessions[session_id]
        language = session.get("language", "en")
        
        print(f"\nDEBUG - Session ID: {session_id}, Language: {language}")
        print(f"DEBUG - Current Stage: {session['stage']}")
        
        # Language-specific questions
        questions_i18n = {
            "en": {
                "age": "What is the age of the goat (in years)?",
                "duration": "How many days has the goat been showing these symptoms?",
                "age_error": "Please enter a valid age (positive number):",
                "duration_error": "Please enter a positive number of days:"
            },
            "hi": {
                "age": "बकरी की उम्र क्या है (वर्षों में)?",
                "duration": "बकरी को ये लक्षण कितने दिनों से दिख रहे हैं?",
                "age_error": "कृपया वैध उम्र दर्ज करें (धनात्मक संख्या):",
                "duration_error": "कृपया धनात्मक दिनों की संख्या दर्ज करें:"
            },
            "mr": {
                "age": "शेळीचे वय किती आहे (वर्षांमध्ये)?",
                "duration": "शेळीला ही लक्षणे किती दिवसांपासून दिसत आहेत?",
                "age_error": "कृपया वैध वय प्रविष्ट करा (धनात्मक संख्या):",
                "duration_error": "कृपया धनात्मक दिवसांची संख्या प्रविष्ट करा:"
            }
        }
        
        q = questions_i18n.get(language, questions_i18n["en"])
        
        # Stage 0: Initial description
        if session["stage"] == "initial":
            session["user_description"] = answer
            session["stage"] = "age"
            return jsonify({
                "question": q["age"],
                "stage": "age"
            })
        
        # Stage 1: Age
        elif session["stage"] == "age":
            try:
                age = float(answer.strip())
                if age < 0:
                    return jsonify({
                        "question": q["age_error"],
                        "stage": "age",
                        "error": "Age must be a positive number"
                    }), 200
                session["age"] = age
            except (ValueError, AttributeError):
                return jsonify({
                    "question": q["age_error"],
                    "stage": "age",
                    "error": "Age must be a number"
                }), 200
            
            session["stage"] = "duration"
            return jsonify({
                "question": q["duration"],
                "stage": "duration"
            })
        
        # Stage 2: Duration
        elif session["stage"] == "duration":
            try:
                duration = int(answer.strip())
                if duration < 0:
                    return jsonify({
                        "question": q["duration_error"],
                        "stage": "duration",
                        "error": "Duration must be a positive number"
                    }), 200
                session["duration_days"] = duration
            except (ValueError, AttributeError):
                return jsonify({
                    "question": q["duration_error"],
                    "stage": "duration",
                    "error": "Duration must be a number"
                }), 200
            
            # Generate counter questions in user's language
            try:
                counter_questions = ask_llm_for_counter_questions(
                    session["user_description"], 
                    session["top3"],
                    language
                )
                session["counter_questions"] = counter_questions
                session["current_question_index"] = 0
                session["stage"] = "questions"
                
                return jsonify({
                    "question": counter_questions[0],
                    "stage": "questions",
                    "question_number": 1,
                    "total_questions": len(counter_questions)
                })
            except Exception as e:
                print(f"ERROR generating questions: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "error": f"Failed to generate questions: {str(e)}"
                }), 500
        
        # Stage 3: Counter questions
        elif session["stage"] == "questions":
            current_idx = session["current_question_index"]
            
            if current_idx >= len(session["counter_questions"]):
                return jsonify({"error": "Question index out of range"}), 400
                
            current_question = session["counter_questions"][current_idx]
            session["user_answers"][current_question] = answer
            
            if current_idx + 1 < len(session["counter_questions"]):
                session["current_question_index"] += 1
                next_question = session["counter_questions"][current_idx + 1]
                return jsonify({
                    "question": next_question,
                    "stage": "questions",
                    "question_number": current_idx + 2,
                    "total_questions": len(session["counter_questions"])
                })
            else:
                # Generate final diagnosis
                session["stage"] = "complete"
                
                try:
                    diagnosis_result = get_final_diagnosis_structured(
                        session["top3"],
                        session["user_answers"],
                        session["duration_days"],
                        session["age"],
                        session["user_description"],
                        language
                    )
                    
                    return jsonify({
                        "stage": "complete",
                        "diagnosis": diagnosis_result
                    })
                except Exception as e:
                    print(f"ERROR generating diagnosis: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({
                        "error": f"Failed to generate diagnosis: {str(e)}"
                    }), 500
        
        else:
            return jsonify({
                "error": f"Invalid stage: {session.get('stage', 'unknown')}"
            }), 400
    
    except Exception as e:
        print(f"ERROR - Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Goat Disease Diagnosis API is running"})

def get_final_diagnosis_structured(top3_diseases, user_answers, duration_days, age, user_description, language="en"):
    """Generate a structured veterinary diagnosis in the specified language."""
    
    top1_disease = top3_diseases[0]["disease"]
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])

    # Healthy / Invalid Handling
    if top1_disease in ["HealthyGoatAug", "InvalidGoatAug"]:
        healthy_messages = {
            "en": {
                "disease": "Healthy" if top1_disease == "HealthyGoatAug" else "Invalid Image",
                "cause": "No disease detected. The goat appears healthy based on the image analysis." if top1_disease == "HealthyGoatAug" else "Unable to analyze - image may be unclear or not showing goat skin condition.",
                "precautions": "Continue regular health monitoring. Maintain clean living conditions. Ensure proper nutrition and vaccination schedule. Regular deworming as per veterinary advice.",
                "care": "Provide clean water and nutritious feed daily. Maintain proper ventilation in housing. Regular grooming and hoof care. Monitor for any changes in behavior or appetite.",
                "home": "N/A - No treatment needed for healthy goat. Focus on preventive care and nutrition.",
                "treatment": "No treatment required. Continue routine veterinary check-ups and maintain vaccination schedule."
            },
            "hi": {
                "disease": "स्वस्थ (Healthy)" if top1_disease == "HealthyGoatAug" else "अमान्य छवि (Invalid Image)",
                "cause": "कोई बीमारी नहीं पाई गई। छवि विश्लेषण के आधार पर बकरी स्वस्थ प्रतीत होती है।" if top1_disease == "HealthyGoatAug" else "विश्लेषण करने में असमर्थ - छवि अस्पष्ट हो सकती है या बकरी की त्वचा की स्थिति नहीं दिखा रही है।",
                "precautions": "नियमित स्वास्थ्य निगरानी जारी रखें। स्वच्छ रहने की स्थिति बनाए रखें। उचित पोषण और टीकाकरण कार्यक्रम सुनिश्चित करें। पशु चिकित्सक की सलाह के अनुसार नियमित रूप से कृमि मुक्ति करें।",
                "care": "प्रतिदिन स्वच्छ पानी और पौष्टिक चारा प्रदान करें। आवास में उचित वेंटिलेशन बनाए रखें। नियमित सौंदर्य और खुर की देखभाल। व्यवहार या भूख में किसी भी बदलाव की निगरानी करें।",
                "home": "लागू नहीं - स्वस्थ बकरी के लिए उपचार की आवश्यकता नहीं है। निवारक देखभाल और पोषण पर ध्यान दें।",
                "treatment": "कोई उपचार आवश्यक नहीं। नियमित पशु चिकित्सा जांच जारी रखें और टीकाकरण कार्यक्रम बनाए रखें।"
            },
            "mr": {
                "disease": "निरोगी (Healthy)" if top1_disease == "HealthyGoatAug" else "अवैध प्रतिमा (Invalid Image)",
                "cause": "कोणताही रोग आढळला नाही. प्रतिमा विश्लेषणावर आधारित शेळी निरोगी दिसते." if top1_disease == "HealthyGoatAug" else "विश्लेषण करण्यास अक्षम - प्रतिमा अस्पष्ट असू शकते किंवा शेळीच्या त्वचेची स्थिती दर्शवत नाही.",
                "precautions": "नियमित आरोग्य निरीक्षण सुरू ठेवा. स्वच्छ राहणीमान राखा. योग्य पोषण आणि लसीकरण वेळापत्रक सुनिश्चित करा. पशुवैद्यकीय सल्ल्यानुसार नियमित जंतुनाशक करा.",
                "care": "दररोज स्वच्छ पाणी आणि पौष्टिक खाद्य पुरवा. घरात योग्य वायुवीजन राखा. नियमित ग्रूमिंग आणि खुराची काळजी. वर्तन किंवा भूक मध्ये कोणत्याही बदलाचे निरीक्षण करा.",
                "home": "लागू नाही - निरोगी शेळीसाठी उपचाराची आवश्यकता नाही. प्रतिबंधात्मक काळजी आणि पोषणावर लक्ष केंद्रित करा.",
                "treatment": "कोणतेही उपचार आवश्यक नाही. नियमित पशुवैद्यकीय तपासणी सुरू ठेवा आणि लसीकरण वेळापत्रक राखा."
            }
        }
        
        msg = healthy_messages.get(language, healthy_messages["en"])
        
        return {
            "Animal_Type": "Goat",
            "Sub_Type": "Common",
            "Age": age,
            "Illness_Duration_Days": duration_days,
            "Detected_Disease": msg["disease"],
            "Severity": "N/A",
            "Cause_of_Disease": msg["cause"],
            "Precautions": msg["precautions"],
            "Care": msg["care"],
            "Home_Remedies": msg["home"],
            "Treatment_or_Medicine": msg["treatment"],
            "Reported_Symptoms": extract_symptoms_from_description(user_description, user_answers)
        }

    # For actual disease predictions
    final_disease = top1_disease
    severity_dict = goat_skin_diseases.get(final_disease, {}).get("severity_progression", {})
    severity = get_severity_for_duration(duration_days, severity_dict)

    prompt = f"""
You are a licensed veterinary specialist providing a detailed diagnostic report.

**IMAGE ANALYSIS:**
A goat image has been analyzed by a CNN model.
The model's top 3 predicted diseases are:
{json.dumps(top3_diseases, indent=2)}

**OWNER'S INFORMATION:**
Goat Age: {age} years
Symptom Duration: {duration_days} days
Initial Description: "{user_description}"

Follow-up Answers:
{json.dumps(user_answers, indent=2, ensure_ascii=False)}

{lang_instruction}

**INSTRUCTIONS:**
Provide a comprehensive diagnosis using the EXACT section headers below.
Each section must contain detailed, actionable information.
For technical terms not in the response language, include English translation in parentheses.

**FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:**

CAUSE:
[Explain the primary cause, pathogen, or etiology in 2-3 complete sentences. Include scientific name if applicable.]

PRECAUTIONS:
[List 4-6 specific preventive measures, one per line. Start each with a bullet point (•) or dash (-)]

CARE:
[Write 4-5 sentences covering: housing requirements, temperature monitoring (mention normal goat temperature: 101.5-103.5°F / 38.6-39.7°C), nutrition, daily health checks (appetite, posture, skin), and isolation if needed.]

HOME_REMEDIES:
[Describe 2-3 safe, traditional remedies or supportive measures in 2-3 sentences. Be specific about preparation and application.]

TREATMENT:
[Write 3-5 sentences covering: specific medicines (generic names), dosage guidance, duration, and when to seek immediate veterinary care. Mention both topical and systemic treatments if applicable.]

SYMPTOMS:
[List 6-8 key clinical signs observed or reported, one per line with bullet points (•) or dashes (-)]

**CRITICAL:** Each section must have substantial content. Do NOT leave any section empty or write "Not specified".
"""
    
    try:
        response = llm.generate_content(prompt)
        llm_text = response.text.strip()
        
        print("\n" + "="*80)
        print("DEBUG - LLM RESPONSE:")
        print("="*80)
        print(llm_text)
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"ERROR in LLM call: {e}")
        llm_text = ""

    # Parse the response
    parsed_data = parse_llm_diagnosis(llm_text)
    
    # Get treatment text with fallback
    treatment_text = get_treatment_text(final_disease, parsed_data.get("treatment", ""), language)
    
    # Get care text with fallback - THIS IS THE KEY FIX
    care_text = get_care_text(final_disease, parsed_data.get("care", ""), language)
    
    # Extract symptoms
    reported_symptoms = extract_symptoms_from_description(user_description, user_answers)
    
    # If LLM provided symptoms, use those instead
    if parsed_data.get("symptoms") and len(parsed_data["symptoms"]) > 0:
        reported_symptoms = parsed_data["symptoms"]

    # Determine if condition is critical
    is_critical = True if severity in ["Moderate", "Critical"] else False
    
    # Final diagnostic report
    diagnosis = {
        "Animal_Type": "Goat",
        "Sub_Type": "Common",
        "Age": age,
        "Illness_Duration_Days": float(duration_days),
        "Detected_Disease": final_disease,
        "Severity": severity,
        "Critical": is_critical,
        "Cause_of_Disease": parsed_data.get("cause", "Not specified"),
        "Precautions": parsed_data.get("precautions", "Not specified"),
        "Care": care_text,  # Using the enhanced care_text function
        "Home_Remedies": parsed_data.get("home_remedies", "Not specified"),
        "Treatment_or_Medicine": treatment_text,
        "Reported_Symptoms": reported_symptoms
    }
    
    print("\n" + "="*80)
    print("DEBUG - FINAL DIAGNOSIS:")
    print("="*80)
    print(f"Care field length: {len(diagnosis['Care'])}")
    print(f"Care content: {diagnosis['Care']}")
    print("="*80 + "\n")
    
    return diagnosis

if __name__ == "__main__":
    print("Starting Goat Disease Diagnosis API on port 5005...")
    app.run(host='0.0.0.0', port=5005, debug=True)
