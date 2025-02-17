from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel
import torch.nn as nn

app = FastAPI()

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

    result = predict_stuttering(file_path)

    return {"status": "success", "detail": "Got the audio successfully", "result": result}
