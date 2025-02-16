let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let dataArray;
let animationId;
const canvas = document.getElementById("wave-canvas");
const canvasCtx = canvas.getContext("2d");

// Adjust canvas dimensions to match its display size
function resizeCanvas() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// Function to draw the waveform
function drawWave() {
    animationId = requestAnimationFrame(drawWave);
    // Get time-domain data for waveform visualization
    analyser.getByteTimeDomainData(dataArray);

    // Clear the canvas
    canvasCtx.fillStyle = "#f0f0f0";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    // Set up the waveform line style
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#0077ff";
    canvasCtx.beginPath();

    const sliceWidth = canvas.width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0; // Normalize between 0 and 2
        const y = (v * canvas.height) / 2;

        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

// Start recording and waveform animation
document.getElementById("start-btn").addEventListener("click", async () => {
    try {
        // Request access to the microphone
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Initialize MediaRecorder for saving the audio
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        // Set up the Web Audio API for waveform visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);

        // Create a buffer to store the time-domain data
        dataArray = new Uint8Array(analyser.fftSize);
        drawWave();

        // Collect audio data as it becomes available
        mediaRecorder.addEventListener("dataavailable", event => {
            audioChunks.push(event.data);
        });

        // When recording stops, create a Blob from the audio chunks and send it to the server
        mediaRecorder.addEventListener("stop", () => {
            cancelAnimationFrame(animationId); // Stop the waveform animation
            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            // Reset chunks for next recording
            audioChunks = [];
            sendAudio(audioBlob);
            // Close the AudioContext if you don't need it further
            if (audioContext && audioContext.state !== "closed") {
                audioContext.close();
            }
        });
    } catch (error) {
        console.error("Microphone access error:", error);
    }
});

// Stop recording
document.getElementById("stop-btn").addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
});

// Function to send audio Blob to the backend
async function sendAudio(blob) {
    const formData = new FormData();
    // Append the blob with a filename (adjust extension/type as needed)
    formData.append("file", blob, "recording.webm");

    try {
        const response = await fetch("http://localhost:8000/api/detect", {
            method: "POST",
            body: formData,
        });
        const result = await response.json();
        console.log("Upload result:", result);
    } catch (error) {
        console.error("Error uploading audio:", error);
    }
}

