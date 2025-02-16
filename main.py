from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/detect")
async def detectStutter(request: Request):
    audioData = await request.body()

    if not audioData:
        raise HTTPException(status_code=400, detail="No audio data received")

    # Temporary will delete later 
    file_path = "audio.wav"
    try:
        with open(file_path, "wb") as file:
            file.write(audioData)
    except Exception as err:
        print(err)
        raise HTTPException(status_code=500, detail="Something unexpected happened!")

    return {"status": "success", "detail": "Got the audio successfully"}
