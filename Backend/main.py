from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel
import torch.nn as nn
from dotenv import load_dotenv
import os
import random
from typing import List, Dict
import json

app = FastAPI()

load_dotenv()
api_key = os.getenv("GEMINI")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# class WhisperForStutteringClassification(nn.Module):
#     def __init__(self, model_name="openai/whisper-small", num_labels=2):
#         super().__init__()
#         self.whisper = WhisperModel.from_pretrained(model_name)
#         self.whisper.encoder.gradient_checkpointing = False
#         self.classifier = nn.Linear(self.whisper.config.d_model, num_labels)
#
#     def forward(self, input_features):
#         encoder_outputs = self.whisper.encoder(input_features).last_hidden_state
#         pooled_output = encoder_outputs.mean(dim=1)
#         logits = self.classifier(pooled_output)
#         return logits
#
# # file path of trained model
# model_path = "/stuttering_detetction/model/whisper-small/stuttering_detection_model.pth"
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model = WhisperForStutteringClassification()
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()
#
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
#
# def preprocess_audio(audio_path):
#     waveform, sample_rate = torchaudio.load(audio_path)
#     if sample_rate != 16000:
#         transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         waveform = transform(waveform)
#     input_features = feature_extractor(
#         waveform.numpy(),
#         sampling_rate=16000,
#         return_tensors="pt"
#     ).input_features
#
#     return input_features.to(device)
#
# def predict_stuttering(audio_path):
#     input_features = preprocess_audio(audio_path)
#
#     with torch.no_grad():
#         logits = model(input_features)
#
#     probabilities = torch.softmax(logits, dim=1)
#     prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
#
#     return "Stuttering Detected" if prediction == 1 else "No Stuttering"

def get_varied_context() -> dict:
    therapist_types = [
        {"personality": "empathetic and nurturing", "experience": "20 years working with children"},
        {"personality": "technical and analytical", "experience": "specializing in adult stuttering"},
        {"personality": "holistic and mindfulness-focused", "experience": "background in psychology"},
        {"personality": "direct and practical", "experience": "expert in fluency techniques"},
        {"personality": "research-oriented", "experience": "focus on neurological factors"}
    ]
    
    approaches = [
        "using the latest research in neurolinguistics",
        "drawing from cognitive behavioral therapy",
        "incorporating mindfulness techniques",
        "using evidence-based fluency shaping",
        "applying the stuttering modification approach"
    ]
    
    perspectives = [
        "Consider the psychological aspects primarily",
        "Focus on the physiological mechanisms",
        "Examine the environmental factors",
        "Analyze the linguistic patterns",
        "Evaluate the social impact"
    ]
    
    time_contexts = [
        "in a first assessment session",
        "during a follow-up evaluation",
        "in an emergency consultation",
        "after six months of therapy",
        "during a group therapy session"
    ]

    return {
        "therapist": random.choice(therapist_types),
        "approach": random.choice(approaches),
        "perspective": random.choice(perspectives),
        "context": random.choice(time_contexts)
    }

def generate_content(stuttering: str, api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Get random context elements
    context = get_varied_context()
    
    prompt = f"""You are a speech therapist who is {context['therapist']['personality']} with {context['therapist']['experience']}.
    {context['perspective']}, {context['approach']}, {context['context']}.
    
    The stuttering detection system has returned: '{stuttering}'
    
    Provide a unique response in exactly 3 paragraphs (80 words each):
    1. First paragraph should be your initial assessment and emotional support
    2. Second paragraph should analyze potential triggers and patterns
    3. Third paragraph should give specific recommendations
    
    Make sure to incorporate your specific therapeutic approach and context into the response.
    Each time you respond, give substantially different advice and perspectives based on your assigned therapeutic style."""
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": round(random.uniform(0.8, 1.0), 2),
            "topP": round(random.uniform(0.85, 0.95), 2),
            "maxOutputTokens": 800,
            "stopSequences": ["4.", "Fourth"]  # Ensure it stops after 3 paragraphs
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
# route for client
@app.post("/api/detect")
async def detect_stutter(file: UploadFile = File(...)):

    file_path = "audio.wav"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something unexpected happened!")

    # result = predict_stuttering(file_path)
    result = "No stuttering"
    feedback = generate_content(result, api_key)
    feedback = feedback["candidates"][0]["content"]["parts"][0]["text"]
    lines = feedback.split('\n')
    data = []
    for i in lines:
        if len(i) != 0:
            data.append(i)


    return {"status": "success", "feedback": data, "result": result}

