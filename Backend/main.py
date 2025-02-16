from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/detect")
async def detect_stutter(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file received")

    file_path = "audio.wav"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something unexpected happened!")

    return {"status": "success", "detail": "Got the audio successfully"}

