# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
import cv2
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize variables as None so the server boots instantly
text_detector = None
image_detector = None

# --- LAZY LOADING FUNCTIONS ---
def get_text_model():
    global text_detector
    if text_detector is None:
        print("Lazy Loading Text Model...")
        text_detector = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
    return text_detector

def get_image_model():
    global image_detector
    if image_detector is None:
        print("Lazy Loading Image Model... (This will take a moment)")
        image_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    return image_detector
# ------------------------------

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Multimodal AI Detector API is running and ports are open!"}

@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    if len(request.text.split()) < 10:
        return {"error": "Please provide at least 10 words."}
    
    # Load the model only when this endpoint is called
    model = get_text_model()
    result = model(request.text, truncation=True, max_length=512)
    
    prediction = result[0]['label'] 
    is_ai = prediction.lower() != "human"
    
    return {
        "is_ai": is_ai,
        "confidence": round(result[0]['score'] * 100, 2),
        "type": "text"
    }

@app.post("/analyze-image")
async def analyze_img(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Load the model only when this endpoint is called
    model = get_image_model()
    results = model(image)
    
    top_result = results[0]
    is_ai = top_result['label'].lower() == "artificial" or "ai" in top_result['label'].lower()
    
    return {
        "is_ai": is_ai,
        "confidence": round(top_result['score'] * 100, 2),
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
        
        # Load the model only when this endpoint is called
        model = get_image_model()
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_converted)
                
                result = model(pil_image)[0]
                is_ai_frame = result['label'].lower() == "artificial" or "ai" in result['label'].lower()
                
                score = result['score'] if is_ai_frame else (1 - result['score'])
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