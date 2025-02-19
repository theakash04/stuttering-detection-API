"use client";
import { Button } from "@/components/ui/button";
import { File, Mic, Square } from "lucide-react";
import React, { useRef, useEffect, useState } from "react";
import { motion } from 'framer-motion';
import TypewriterFeedback from "@/components/typrwritterFeedback";

interface response {
    status: string,
    feedback: [string],
    result: string
}

const AudioRecorder: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const dataArrayRef = useRef<Uint8Array | null>(null);
    const animationIdRef = useRef<number>(0);
    const streamRef = useRef<MediaStream | null>(null);
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [showResult, setShowResult] = useState<response | null>(null)
    const [isProcessing, setIsProcessing] = useState<boolean>(false)
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [isMicPermission, setIsMicPermission] = useState<boolean>(false);
    const [isProceed, setIsProceed] = useState<boolean>(false)
    const [selectedFile, setSelectedFile] = useState<any | null>(null);

    // Adjust canvas dimensions and draw a placeholder background on load
    const resizeCanvas = () => {
        if (canvasRef.current) {
            canvasRef.current.width = 700;
            canvasRef.current.height = 200;
        }
    };

    // Draw a placeholder dark background
    const drawPlaceholder = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) {
                ctx.fillStyle = "rgb(10, 10, 10)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
        }
    };

    useEffect(() => {
        resizeCanvas();
        drawPlaceholder();
        window.addEventListener("resize", resizeCanvas);

        // Ask for microphone permission on load and store the stream.
        const initAudio = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                streamRef.current = stream;
                setIsMicPermission(true)
            } catch (error) {
                console.error("Permission denied for audio", error);
                setIsMicPermission(false)
            }
        };
        initAudio();

        return () => {
            window.removeEventListener("resize", resizeCanvas);
            if (audioContextRef.current && audioContextRef.current.state !== "closed") {
                audioContextRef.current.close();
            }
            cancelAnimationFrame(animationIdRef.current);
        };
    }, []);

    // Draw frequency data as bars
    const drawFrequency = () => {
        const canvas = canvasRef.current;
        if (!canvas || !analyserRef.current || !dataArrayRef.current) return;
        const canvasCtx = canvas.getContext("2d");
        if (!canvasCtx) return;

        // Get frequency-domain data
        analyserRef.current.getByteFrequencyData(dataArrayRef.current);

        // Clear the canvas
        canvasCtx.fillStyle = "rgb(10, 10, 10)";
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw frequency bars
        const barWidth = (canvas.width / dataArrayRef.current.length) * 2;
        let x = 0;
        for (let i = 0; i < dataArrayRef.current.length; i++) {
            const barHeight = dataArrayRef.current[i];
            canvasCtx.fillStyle = "rgb(125, 211, 252)";
            canvasCtx.fillRect(x, canvas.height - barHeight / 1.2, barWidth, barHeight / 1.2);
            x += barWidth + 1;
        }

        animationIdRef.current = requestAnimationFrame(drawFrequency);
    };

    // Send the recorded audio Blob to your backend
    const sendAudio = async (blob: Blob) => {
        const formData = new FormData();
        formData.append("file", blob, "recording.wav");

        setIsProcessing(true);

        try {
            // Now proceed with the API call
            const response = await fetch("http://localhost:8000/api/detect", {
                method: "POST",
                body: formData,
            });
            const result = await response.json();
            setShowResult(result);
        } catch (error) {
            console.error("Error uploading audio:", error);
        } finally {
            setIsProcessing(false);
            setIsProceed(true)
        }
    };

    // Starts the recording process using the stored stream
    const startRecording = (stream: MediaStream) => {
        try {
            // Initialize MediaRecorder for recording the audio stream
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.start();

            // Set up the Web Audio API for frequency visualization
            const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
            const audioContext = new AudioContext();
            audioContextRef.current = audioContext;
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);
            analyserRef.current = analyser;

            // Create a buffer to store the frequency data
            dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
            drawFrequency();

            // Collect audio data as it becomes available
            mediaRecorder.addEventListener("dataavailable", (event) => {
                audioChunksRef.current.push(event.data);
            });

            // When recording stops, send the audio blob to the backend and create a URL for playback
            mediaRecorder.addEventListener("stop", async () => {
                cancelAnimationFrame(animationIdRef.current);
                const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
                audioChunksRef.current = []; // Reset for next recording
                const newAudioUrl = URL.createObjectURL(audioBlob);
                setAudioUrl(newAudioUrl);
                await sendAudio(audioBlob);
                if (audioContextRef.current && audioContextRef.current.state !== "closed") {
                    audioContextRef.current.close();
                }
            });
            setIsRecording(true);
        } catch (error) {
            console.error("Error during recording:", error);
        }
    };
    // Handle button click to start/stop recording
    const handleMicButton = async () => {
        if (!isRecording) {
            // Use the stream obtained on load or ask permission again if not available
            const stream = streamRef.current ?? (await navigator.mediaDevices.getUserMedia({ audio: true }));
            streamRef.current = stream;
            startRecording(stream);
        } else {
            // Stop the recording
            if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
                mediaRecorderRef.current.stop();
            }
            setIsRecording(false);
        }
    };

    const handleFileUpload = async (event: any) => {
        const file = event.target.files[0];
        if (file.type !== "audio/wav") {
            console.error("Please upload a WAV file.");
            return;
        }
        const fileUrl = URL.createObjectURL(file);
        setSelectedFile(file);
        setAudioUrl(fileUrl);
        await sendAudio(file);

    };


    return (
        <div className="w-screen min-h-screen flex items-center">
            {!isProcessing && !showResult && (
                <div className="w-screen flex flex-col items-center justify-center gap-10 min-h-screen">
                    {isMicPermission ? (<canvas
                        id="wave-canvas"
                        ref={canvasRef}
                        className=""
                        style={{ width: "700px", height: "200px" }}
                    />) : (
                        <div className="text-destructive font-semibold ">
                            Microphone access is disabled please enable it in your settings to continue.
                        </div>
                    )}
                    <div className="flex items-center justify-center w-full gap-5 flex-col">
                        <Button
                            id="start-btn"
                            onClick={handleMicButton}
                            type="button"
                            disabled={!isMicPermission}
                            size={"lg"}
                            className={`${isMicPermission ? "bg-red-500" : "bg-slate-300/30"} w-28 h-28 rounded-full hover:bg-red-600 border-2 border-muted-foreground`}
                        >
                            {isRecording ? <Square color="white" size={"38px"} /> : <Mic color="white" size={"38px"} />}
                        </Button>
                        <div className="mt-8 flex flex-col items-center">
                            <input
                                type="file"
                                accept=".wav"
                                onChange={handleFileUpload}
                                className="hidden"
                                id="file-upload"
                            />

                            <label
                                htmlFor="file-upload"
                                className="flex flex-col items-center justify-center cursor-pointer border-2 border-slate-600 rounded-lg p-6 transition-all hover:border-sky-100 hover:bg-slate-800/40 group w-64"
                            >
                                <div className="mb-3 text-sky-100 group-hover:text-sky-100 transition-colors">
                                    <svg
                                        className="w-8 h-8"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth="2"
                                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                                        />
                                    </svg>
                                </div>

                                <p className="text-center text-muted-foreground group-hover:text-sky-100 transition-colors">
                                    <span className="font-medium text-sky-100">Browse WAV files</span>
                                    <span className="block text-sm mt-1 text-slate-400">Max 20MB</span>
                                </p>
                            </label>

                            {selectedFile && (
                                <p className="mt-3 text-sm text-emerald-400 flex items-center gap-1">
                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                    </svg>
                                    {selectedFile.name}
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            )
            }
            {
                isProcessing && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="space-y-4 w-10/12 mx-auto"
                    >
                        <div className="relative w-full bg-transparent rounded-full h-4 overflow-hidden flex items-center justify-center">
                            {/* Animated shimmer overlay */}
                            <div className="absolute inset-0">
                                <div className="absolute h-full w-1/2 bg-gradient-to-r from-transparent via-blue-400/30 to-transparent animate-shimmer" />
                            </div>
                        </div>
                        <div className="text-center text-gray-400">
                            Processing...
                        </div>
                    </motion.div>
                )
            }

            {
                isProceed && audioUrl && showResult && (
                    <div className="grid grid-cols-1 place-content-center w-screen min-h-screen bg-background p-8">
                        <h2 className="text-3xl font-bold mb-6 text-sky-100 text-center animate-fade-in">
                            Analysis Results
                            <span className="block w-16 h-1 bg-sky-500 mt-2 mx-auto rounded-full" />
                        </h2>

                        <div className="flex flex-col gap-6 items-center mb-8">
                            <motion.div
                                initial={{ scale: 0.9, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                transition={{ duration: 0.4 }}
                                className="w-full max-w-96 flex items-center justify-center flex-col"
                            >
                                <audio
                                    controls
                                    src={audioUrl}
                                    className="custom-audio transition-shadow bg-background w-8/12"
                                >
                                    Your browser does not support the audio element.
                                </audio>
                            </motion.div>

                            <div className="space-y-6 w-full max-w-2xl">
                                <motion.div
                                    className="bg-slate-700/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-sky-200/30"
                                    initial={{ y: 20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                >
                                    <div className="flex items-center gap-3 mb-4">
                                        <img className="text-2xl w-7 h-7" src="target.svg" />
                                        <p className="text-xl font-semibold text-sky-100">Analysis Result</p>
                                    </div>
                                    <p className="text-muted-foreground leading-relaxed">
                                        {showResult.result}
                                    </p>
                                </motion.div>

                                <motion.div
                                    className="bg-slate-700/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-sky-200/30"
                                    initial={{ y: 20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ delay: 0.1 }}
                                >
                                    <div className="flex items-center gap-3 mb-4">
                                        <img className="text-2xl w-7 h-7" src="bulb.svg" />
                                        <p className="text-xl font-semibold text-sky-100">Feedback</p>
                                    </div>
                                    <div className="text-muted-foreground leading-relaxed">
                                        <div>
                                            <TypewriterFeedback feedback={showResult.feedback} />
                                        </div>
                                    </div>
                                </motion.div>
                            </div>
                        </div>

                        <motion.div
                            className="flex items-center justify-center"
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <Button
                                size="lg"
                                className="rounded-lg bg-sky-100 hover:bg-white text-black shadow-md hover:shadow-lg transition-all duration-300 px-8 py-6 text-lg font-semibold"
                                onClick={() => { setIsProcessing(false); setShowResult(null); setIsProceed(false); setSelectedFile(null) }}
                            >

                                Start New Recording
                            </Button>
                        </motion.div>
                    </div>
                )
            }
        </div >
    );
};

export default AudioRecorder;

