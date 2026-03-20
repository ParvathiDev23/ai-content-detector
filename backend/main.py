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

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Hugging Face API Endpoints
TEXT_API_URL = "https://router.huggingface.co/hf-inference/models/Hello-SimpleAI/chatgpt-detector-roberta"

# FIX 1: Swapped to a highly active, modern Deepfake Detector model
IMAGE_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class TextRequest(BaseModel):
    text: str

def query_huggingface(api_url, payload, is_file=False):
    max_retries = 3
    for attempt in range(max_retries):
        if is_file:
            # FIX 2: We must explicitly tell Hugging Face this is raw binary image data
            file_headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream"
            }
            response = requests.post(api_url, headers=file_headers, data=payload)
        else:
            response = requests.post(api_url, headers=HEADERS, json=payload)
            
        result = response.json()
        
        if isinstance(result, dict) and "estimated_time" in result:
            wait_time = result["estimated_time"]
            print(f"Model waking up. Waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
            
        return result
    return {"error": "Hugging Face API timed out or failed"}

@app.get("/")
def read_root():
    return {"status": "Lightweight API is running perfectly!"}

@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    if len(request.text.split()) < 10:
        return {"error": "Please provide at least 10 words."}
    
    result = query_huggingface(TEXT_API_URL, {"inputs": request.text})
    if "error" in result: return result
        
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
    
    if isinstance(result, dict) and "error" in result:
        return result
        
    top_prediction = result[0]
    label = top_prediction['label'].lower()
    
    # Catching multiple label types (Fake, Deepfake, Artificial, etc.)
    is_ai = any(word in label for word in ["fake", "artificial", "ai", "deepfake"])
    
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
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    result = query_huggingface(IMAGE_API_URL, buffer.tobytes(), is_file=True)
                    if isinstance(result, list):
                        top_prediction = result[0]
                        label = top_prediction['label'].lower()
                        is_ai_frame = any(word in label for word in ["fake", "artificial", "ai", "deepfake"])
                        score = top_prediction['score'] if is_ai_frame else (1 - top_prediction['score'])
                        ai_scores.append(score)
                
        cap.release()
        
        if not ai_scores:
            return {"error": "Could not extract or analyze frames from video."}
            
        avg_ai_score = sum(ai_scores) / len(ai_scores)
        is_ai_overall = avg_ai_score > 0.5

        return {
            "is_ai": is_ai_overall,
            "confidence": round((avg_ai_score if is_ai_overall else (1 - avg_ai_score)) * 100, 2),
            "type": "video"
        }
    finally:
        os.remove(temp_video_path)