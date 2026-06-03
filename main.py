from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import os
import re
import json
import tempfile
import subprocess
import datetime
import textwrap

import speech_recognition as sr

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from dotenv import load_dotenv
from groq import Groq

from langdetect import detect
from deep_translator import GoogleTranslator

from pymongo import MongoClient

# ============================================
# LOAD ENV
# ============================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URL = os.getenv("MONGO_URL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing")

if not MONGO_URL:
    raise ValueError("MONGO_URL missing")

# ============================================
# GROQ CLIENT
# ============================================

client = Groq(api_key=GROQ_API_KEY)

# ============================================
# MONGODB
# ============================================

mongo_client = MongoClient(MONGO_URL)
db = mongo_client["ai_clinical_assistant"]
prescriptions_collection = db["prescriptions"]
transcriptions_collection = db["transcriptions"]

# ============================================
# FASTAPI
# ============================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# FONT SETUP
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "Font", "NotoSansDevanagari-Regular.ttf")
FONT_NAME = "Helvetica"

try:
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont("UnicodeFont", FONT_PATH))
        FONT_NAME = "UnicodeFont"
except:
    FONT_NAME = "Helvetica"

# ============================================
# SAFE JSON
# ============================================

def safe_json_load(text):
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

# ============================================
# DETECT LANGUAGE
# ============================================

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ============================================
# TRANSLATE TO ENGLISH
# ============================================

def translate_to_english(text):
    try:
        lang = detect_language(text)
        if lang == "en":
            return text
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated
    except:
        return text

# ============================================
# TRANSLATE FROM ENGLISH
# ============================================

def translate_from_english(text, target_lang):
    try:
        if target_lang == "en":
            return text
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        return translated
    except:
        return text

# ============================================
# CLEAN TEXT
# ============================================

def clean_ai_text(text):
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    return text.strip()

# ============================================
# NATURAL ENGLISH
# ============================================

def make_natural_english(text):
    try:
        prompt = f"""
Convert this patient statement into clean natural English.

ONLY RETURN FINAL ENGLISH SENTENCE.

Patient Statement:
{text}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        final_text = response.choices[0].message.content.strip()
        return clean_ai_text(final_text)
    except:
        return text

# ============================================
# DRAW TEXT
# ============================================

def draw_multiline_text(c, text, x, y, width=80, font_name="Helvetica", font_size=11):
    c.setFont(font_name, font_size)
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            y -= 10
            continue
        wrapped = textwrap.wrap(line, width=width)
        for wrap_line in wrapped:
            c.drawString(x, y, wrap_line)
            y -= 15
            if y < 80:
                c.showPage()
                c.setFont(font_name, font_size)
                y = 750
    return y

# ============================================
# AUDIO TRANSCRIPTION
# ============================================

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    webm_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await file.read())
            webm_path = tmp.name

        wav_path = webm_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        language_options = [
            ("mr-IN", "mr"), ("hi-IN", "hi"), ("gu-IN", "gu"),
            ("ta-IN", "ta"), ("te-IN", "te"), ("kn-IN", "kn"),
            ("bn-IN", "bn"), ("ml-IN", "ml"), ("pa-IN", "pa"),
            ("en-IN", "en")
        ]

        original_text = ""
        detected_language = "en"

        for speech_lang, short_lang in language_options:
            try:
                text = recognizer.recognize_google(audio, language=speech_lang)
                if text and len(text.strip()) > 2:
                    original_text = text
                    detected_language = short_lang
                    break
            except:
                continue

        if not original_text:
            original_text = recognizer.recognize_google(audio)
            detected_language = detect_language(original_text)

        translated_text = translate_to_english(original_text)
        clean_english = make_natural_english(translated_text)

        transcriptions_collection.insert_one({
            "original_text": original_text,
            "english_text": clean_english,
            "language": detected_language,
            "created_at": datetime.datetime.utcnow()
        })

        return {
            "original_text": original_text,
            "text": clean_english,
            "language": detected_language
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for path in [webm_path, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)

# ============================================
# ANALYZE TEXT
# ============================================

@app.post("/analyze_text")
async def analyze_text(req: Request):
    try:
        data = await req.json()
        text = data.get("text", "").strip()

        if not text:
            return JSONResponse(status_code=400, content={"error": "No text provided"})

        detected_language = detect_language(text)
        translated_text = translate_to_english(text)
        translated_text = make_natural_english(translated_text)

        prompt = f"""
You are a professional medical AI assistant.

Analyze patient symptoms.

Transcript:
{translated_text}

Return ONLY VALID JSON.

FORMAT:
{{
  "cleaned_text": "clean sentence",
  "summary": "short summary",
  "entities": [
    {{
      "type": "symptom",
      "value": "cold",
      "confidence": 0.95
    }}
  ],
  "suggested_questions": [
    "How long have symptoms existed?"
  ]
}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )

        raw = response.choices[0].message.content
        parsed = safe_json_load(raw)

        if not parsed:
            return JSONResponse(status_code=500, content={"error": "Invalid AI response"})

        return {
            "language": detected_language,
            "translated_text": translated_text,
            "cleaned_text": parsed.get("cleaned_text", translated_text),
            "summary": parsed.get("summary", ""),
            "entities": parsed.get("entities", []),
            "suggested_questions": parsed.get("suggested_questions", [])
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================
# GENERATE PRESCRIPTION
# ============================================

@app.post("/generate_prescription")
async def generate_prescription(req: Request):
    try:
        data = await req.json()
        doctor = data.get("doctor")
        patient = data.get("name")
        age = data.get("age")
        symptoms = data.get("symptoms", "")
        entities = data.get("entities", [])

        if not doctor or not patient or not age:
            return JSONResponse(status_code=400, content={"error": "Missing required fields"})

        today = datetime.date.today().strftime("%d-%m-%Y")
        detected_language = detect_language(symptoms)
        symptoms_en = translate_to_english(symptoms)
        symptoms_en = make_natural_english(symptoms_en)

        entity_text = ""
        for ent in entities:
            entity_text += f"- {ent.get('type')} : {ent.get('value')}\n"

        prompt = f"""
You are a professional clinical prescription assistant.

IMPORTANT:
The doctor already edited and finalized the prescription content.

Your job is ONLY to:
- organize it professionally
- preserve doctor's intent
- preserve same medicines
- preserve same symptoms
- preserve same diagnosis
- preserve same tests
- preserve same advice

DO NOT:
- add extra medicines
- add extra symptoms
- add extra tests
- add extra disease names
- add unnecessary recommendations
- overexplain
- generate long AI style prescriptions
- change doctor's prescription meaning

VERY IMPORTANT RULES:
1. If medicines are already mentioned in symptoms/transcript, DO NOT replace them.
2. If doctor mentions "no tests", then write: Tests: - None
3. If doctor gives only one medicine, keep only one medicine.
4. If doctor gives short prescription, keep output short.
5. DO NOT generate a fixed template every time.
6. Keep prescription natural and doctor-like.
7. If NO medicine is mentioned, then suggest simple common medicines carefully.
8. If dosage is missing, you may add small basic dosage carefully.
9. Keep response concise and realistic.
10. No markdown stars. No ###. No AI wording.

Doctor: {doctor}
Patient: {patient}
Age: {age}
Doctor Edited Prescription / Transcript: {symptoms_en}
Entities: {entity_text}

OUTPUT FORMAT:
Diagnosis:
<diagnosis only if present>

Symptoms:
- symptom

Medicines:
- medicine | dosage | duration

Tests:
- only if mentioned

Advice:
- only if mentioned

Follow-Up:
- only if mentioned
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )

        prescription_english = response.choices[0].message.content.strip()
        prescription_english = clean_ai_text(prescription_english)

        # Better native language prescription
        try:
            language_map = {
                "mr": "Marathi", "hi": "Hindi", "gu": "Gujarati",
                "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
                "bn": "Bengali", "ml": "Malayalam", "pa": "Punjabi", "en": "English"
            }
            target_language_name = language_map.get(detected_language, "English")

            if detected_language == "en":
                native_prescription = prescription_english
            else:
                translation_prompt = f"""
Translate the following medical prescription into proper {target_language_name} language.

IMPORTANT RULES:
- Use ONLY real native script
- Do NOT use English letters
- Marathi must be in देवनागरी
- Hindi must be in देवनागरी
- Gujarati in ગુજરાતી script
- Tamil in தமிழ் script
- Telugu in తెలుగు script
- Keep medicine names same if needed
- Keep formatting same
- Keep medical meaning accurate
- Do not shorten anything
- Do not add extra information

Prescription:
{prescription_english}
"""
                native_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": translation_prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                native_prescription = native_response.choices[0].message.content.strip()
                native_prescription = clean_ai_text(native_prescription)
        except Exception as e:
            print("NATIVE TRANSLATION ERROR:", str(e))
            native_prescription = prescription_english

        english_footer = f"""
Prescribed By: {doctor}
Patient Name: {patient}
Age: {age}
Date: {today}
"""
        native_footer = translate_from_english(
            f"Prescribed By: Dr. {doctor}\nPatient Name: {patient}\nAge: {age}\nDate: {today}",
            detected_language
        )

        prescription_english += english_footer
        native_prescription += native_footer

        prescriptions_collection.insert_one({
            "doctor": doctor,
            "patient": patient,
            "age": age,
            "language": detected_language,
            "symptoms_original": symptoms,
            "symptoms_english": symptoms_en,
            "english_prescription": prescription_english,
            "native_prescription": native_prescription,
            "created_at": datetime.datetime.utcnow()
        })

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", patient)
        pdf_name = f"prescription_{safe_name}.pdf"
        pdf_path = os.path.join(tempfile.gettempdir(), pdf_name)

        c = canvas.Canvas(pdf_path, pagesize=letter)

        # PAGE 1 ENGLISH
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(300, 780, "AI Medical Prescription")
        c.line(50, 765, 550, 765)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, 735, "Date:")
        c.drawString(60, 715, "Doctor:")
        c.drawString(60, 695, "Patient:")
        c.drawString(400, 695, "Age:")
        c.setFont("Helvetica", 12)
        c.drawString(120, 735, today)
        c.drawString(120, 715, doctor)
        c.drawString(120, 695, patient)
        c.drawString(440, 695, str(age))
        c.line(50, 680, 550, 680)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, 650, "English Prescription")
        draw_multiline_text(c, prescription_english, 60, 620, width=85, font_name="Helvetica", font_size=11)
        c.showPage()

        # PAGE 2 NATIVE LANGUAGE
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(300, 780, "Native Language Prescription")
        c.line(50, 765, 550, 765)
        c.setFont(FONT_NAME, 12)
        c.drawString(60, 735, f"Language: {detected_language}")
        c.line(50, 720, 550, 720)
        draw_multiline_text(c, native_prescription, 60, 690, width=85, font_name=FONT_NAME, font_size=11)
        c.save()

        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_name)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================
# ROOT
# ============================================

@app.get("/")
def root():
    return {"message": "AI Clinical Assistant Backend Running Successfully"}