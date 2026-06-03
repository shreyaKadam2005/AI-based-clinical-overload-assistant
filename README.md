# AI-based-clinical-overload-assistant
AI assistant for doctors

Overview:-
  AI-powered healthcare assistant that converts doctor-patient conversations into structured medical records and generates prescriptions automatically.

Key Features
  Real-time Audio Transcription
  Multilingual Language Detection
  Medical Entity Extraction
  AI-Based Clinical Suggestions
  Automatic Prescription Generation
  Doctor Signature Integration
  Interactive Clinical Dashboard
 
Tech Stack
  Frontend: HTML, CSS, JavaScript
  Backend: FastAPI
  AI Models: Hugging Face Transformers.google translater, langdetect
  Speech Recognition: Whisper
  PDF Generation: ReportLab
  
Benefits
  Reduces doctor documentation workload
  Improves prescription accuracy
  Saves consultation time
  Enhances clinical efficiency

Run Project
 python -m venv .venv
 .venv\Scripts\activate
 pip install -r requirements.txt
 cd backend
 uvicorn app.main:app --reload

 
