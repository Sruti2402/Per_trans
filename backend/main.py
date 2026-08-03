from fastapi import FastAPI
from langdetect import detect
from googletrans import Translator

app = FastAPI()

translator = Translator()

@app.get("/")
def home():
    return {"message": "Backend Working"}

@app.get("/detect")
def language_detect(text: str):
    lang = detect(text)
    return {"language": lang}

@app.get("/translate")
def translate(text: str, lang: str):
    result = translator.translate(text, dest=lang)

    return {
        "original_text": text,
        "translated_text": result.text,
        "target_language": lang
    }