from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from bs4 import BeautifulSoup
import requests
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Setup CORS
origins = os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5500,http://localhost:5500').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapeRequest(BaseModel):
    url: HttpUrl
    follow_redirects: bool = True

@app.post("/scrape")
def scrape_website(req: ScrapeRequest):
    try:
        response = requests.get(req.url, allow_redirects=req.follow_redirects, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Get visible text
    text = soup.get_text(separator="\n", strip=True)

    return {"text": text}
