from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel
import torch.nn as nn
from dotenv import load_dotenv
import os

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
class WhisperForStutteringClassification(nn.Module):
    def __init__(self, model_name="openai/whisper-small", num_labels=2):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.whisper.encoder.gradient_checkpointing = False
        self.classifier = nn.Linear(self.whisper.config.d_model, num_labels)

    def forward(self, input_features):
        encoder_outputs = self.whisper.encoder(input_features).last_hidden_state
        pooled_output = encoder_outputs.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

# file path of trained model
model_path = "/stuttering_detetction/model/whisper-small/stuttering_detection_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForStutteringClassification()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    input_features = feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    return input_features.to(device)

def predict_stuttering(audio_path):
    input_features = preprocess_audio(audio_path)

    with torch.no_grad():
        logits = model(input_features)

    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]

    return "Stuttering Detected" if prediction == 1 else "No Stuttering"


def generate_content(stuttering: str, api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Imagine you're a speech therapist. A stuttering detector returns either 'stuttering detected' or 'No Stuttering'. For 'stuttering detected', provide empathetic feedback, insights on possible triggers, and personalized recommendations to manage stuttering. For 'No Stuttering', offer positive reinforcement and tips to maintain fluent speech. give your response in 3 paragraph 80 words each. {stuttering}"
                     }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        # In case of an error, return a dictionary with error details
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

