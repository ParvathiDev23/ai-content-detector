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

# API Endpoints
TEXT_API_URL = "https://router.huggingface.co/hf-inference/models/Hello-SimpleAI/chatgpt-detector-roberta"
# Swapped to a highly stable model to avoid the inverted label bug
IMAGE_API_URL = "https://router.huggingface.co/hf-inference/models/dima806/deepfake_vs_real_image_detection"

class TextRequest(BaseModel):
    text: str

def query_huggingface(api_url, payload, is_file=False):
    max_retries = 3
    for attempt in range(max_retries):
        if is_file:
            file_headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream"
            }
            response = requests.post(api_url, headers=file_headers, data=payload)
        else:
            response = requests.post(api_url, headers=HEADERS, json=payload)
            
        result = response.json()
        if isinstance(result, dict) and "estimated_time" in result:
            time.sleep(result["estimated_time"])
            continue
        return result
    return {"error": "API timed out"}

def parse_hf_response(result):
    """Bulletproof parser to find the exact AI probability, preventing 'all fake' bugs."""
    if isinstance(result, dict) and "error" in result:
        return 0.5
        
    # Handle both nested lists [[{...}]] and flat lists [{...}]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        predictions = result[0]
    elif isinstance(result, list):
        predictions = result
    else:
        return 0.5

    ai_score = 0.0
    human_score = 0.0

    # Dynamically scan for labels rather than trusting the array order
    for pred in predictions:
        label = pred.get('label', '').lower()
        score = pred.get('score', 0.0)

        if any(w in label for w in ["fake", "ai", "artificial", "chatgpt", "deepfake", "machine", "1"]):
            ai_score += score
        elif any(w in label for w in ["human", "real", "authentic", "original", "0"]):
            human_score += score

    # Calculate the final mathematical probability safely
    if ai_score == 0.0 and human_score > 0.0:
        return 1.0 - human_score
    elif human_score == 0.0 and ai_score > 0.0:
        return ai_score
    elif ai_score > 0 and human_score > 0:
        return ai_score / (ai_score + human_score)
        
    return 0.5 # Unknown error fallback

def get_threshold_result(ai_probability, content_type):
    if ai_probability >= 0.80:
        return {"status": "Highly Likely AI-Generated", "level": "danger", "confidence": round(ai_probability * 100, 2), "type": content_type}
    elif ai_probability >= 0.55:
        return {"status": "Mixed Content / Inconclusive", "level": "warning", "confidence": round(ai_probability * 100, 2), "type": content_type}
    else:
        human_prob = 1.0 - ai_probability
        return {"status": "Likely Authentic / Human", "level": "success", "confidence": round(human_prob * 100, 2), "type": content_type}

@app.get("/")
def read_root():
    return {"status": "Bulletproof API is running!"}

@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    if len(request.text.split()) < 10:
        return {"error": "Please provide at least 10 words."}
    
    result = query_huggingface(TEXT_API_URL, {"inputs": request.text})
    if isinstance(result, dict) and "error" in result: return result
        
    ai_prob = parse_hf_response(result)
    return get_threshold_result(ai_prob, "text")

@app.post("/analyze-image")
async def analyze_img(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    result = query_huggingface(IMAGE_API_URL, image_bytes, is_file=True)
    if isinstance(result, dict) and "error" in result: return result
        
    ai_prob = parse_hf_response(result)
    return get_threshold_result(ai_prob, "image")

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
                    ai_prob = parse_hf_response(result)
                    if ai_prob != 0.5: # Skip broken frames
                        ai_scores.append(ai_prob)
        cap.release()
        
        if not ai_scores: return {"error": "Could not analyze video frames."}
        avg_ai_score = sum(ai_scores) / len(ai_scores)
        return get_threshold_result(avg_ai_score, "video")
    finally:
        os.remove(temp_video_path)