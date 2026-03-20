# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import io
import cv2
import tempfile
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grab the API token from Render's environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Hugging Face API Endpoints
TEXT_API_URL = "https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta"
IMAGE_API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"

class TextRequest(BaseModel):
    text: str

def query_huggingface(api_url, payload, is_file=False):
    """Helper function to send data to Hugging Face with retry logic for sleeping models"""
    max_retries = 3
    for attempt in range(max_retries):
        if is_file:
            response = requests.post(api_url, headers=HEADERS, data=payload)
        else:
            response = requests.post(api_url, headers=HEADERS, json=payload)
            
        result = response.json()
        
        # If the HF model is asleep, it returns an "estimated_time" to wake up
        if isinstance(result, dict) and "estimated_time" in result:
            wait_time = result["estimated_time"]
            print(f"Model is waking up. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue
            
        return result
    return {"error": "Hugging Face API timed out"}

@app.get("/")
def read_root():
    return {"status": "Lightweight API is running perfectly!"}

@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    if len(request.text.split()) < 10:
        return {"error": "Please provide at least 10 words."}
    
    result = query_huggingface(TEXT_API_URL, {"inputs": request.text})
    
    if "error" in result:
        return result
        
    # HF returns a list of lists: [[{'label': 'Human', 'score': 0.99}, ...]]
    top_prediction = result[0][0]
    is_ai = top_prediction['label'].lower() != "human"
    
    return {
        "is_ai": is_ai,
        "confidence": round(top_prediction['score'] * 100, 2),
        "type": "text"
    }

@app.post("/analyze-image")
async def analyze_img(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    result = query_huggingface(IMAGE_API_URL, image_bytes, is_file=True)
    
    if "error" in result:
        return result
        
    top_prediction = result[0]
    is_ai = top_prediction['label'].lower() == "artificial" or "ai" in top_prediction['label'].lower()
    
    return {
        "is_ai": is_ai,
        "confidence": round(top_prediction['score'] * 100, 2),
        "type": "image"
    }

@app.post("/analyze-video")
async def analyze_vid(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(total_frames * 0.25), int(total_frames * 0.5), int(total_frames * 0.75)]
        
        ai_scores = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert frame to bytes to send to API
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    result = query_huggingface(IMAGE_API_URL, buffer.tobytes(), is_file=True)
                    if isinstance(result, list):
                        top_prediction = result[0]
                        is_ai_frame = top_prediction['label'].lower() == "artificial" or "ai" in top_prediction['label'].lower()
                        score = top_prediction['score'] if is_ai_frame else (1 - top_prediction['score'])
                        ai_scores.append(score)
                
        cap.release()
        
        if not ai_scores:
            return {"error": "Could not extract frames from video."}
            
        avg_ai_score = sum(ai_scores) / len(ai_scores)
        is_ai_overall = avg_ai_score > 0.5

        return {
            "is_ai": is_ai_overall,
            "confidence": round((avg_ai_score if is_ai_overall else (1 - avg_ai_score)) * 100, 2),
            "type": "video"
        }
    finally:
        os.remove(temp_video_path)