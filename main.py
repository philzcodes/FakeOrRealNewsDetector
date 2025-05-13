from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging
import os

# Initialize logging
tlogging = logging.getLogger('uvicorn.error')
logging.basicConfig(level=logging.INFO)

# Ensure NLTK data is available: Run these once before deploying
# nltk.download('stopwords')
# nltk.download('wordnet')

# Text preprocessing setup
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    tlogging.error("NLTK resource not found. Make sure stopwords and wordnet corpora are downloaded.")
    raise

def clean_text(text: str) -> str:
    # Lowercase and remove URLs
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove non-letters
    text = re.sub(r'[^a-z ]', '', text)
    # Simple whitespace tokenization (avoiding NLTK punkt dependency)
    tokens = text.split()
    # Lemmatize and remove stopwords
    cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(cleaned)

# Load trained model
t_model_path = os.getenv('MODEL_PATH', 'fake_news_detector_model.joblib')
if not os.path.exists(t_model_path):
    tlogging.error(f"Model file not found at {t_model_path}. Please ensure the path is correct.")
    raise FileNotFoundError(f"Model file not found at {t_model_path}")
model = joblib.load(t_model_path)

# FastAPI app
title = "Fake News Detector API"
description = "Upload news text and get a FAKE/REAL prediction powered by a Logistic Regression model."
app = FastAPI(title=title, description=description)

# Setup CORS
origins = os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5500,http://localhost:5500').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict_news(news: NewsInput):
    try:
        tlogging.info(f"Received text of length {len(news.text)} for prediction.")
        cleaned = clean_text(news.text)
        pred = model.predict([cleaned])[0]
        label = "FAKE" if pred == 1 else "REAL"
        tlogging.info(f"Prediction: {label}")
        return {"prediction": label}
    except Exception as e:
        tlogging.exception("Error during prediction:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake News Detector API. POST /predict with your text."}
