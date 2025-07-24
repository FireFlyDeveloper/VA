from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import pvporcupine
import struct
import asyncio
from collections import deque
import numpy as np
import wave
import speech_recognition as sr
import io
import time

app = FastAPI()

# Initialize Porcupine wake word engine
porcupine = pvporcupine.create(
    access_key='YOUR_PINECONE_API_KEY',  # Replace with your real access key
    keyword_paths=['hey_ani.ppn']
)
FRAME_LENGTH = porcupine.frame_length
SAMPLE_RATE = porcupine.sample_rate

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = deque(maxlen=FRAME_LENGTH * 10)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = struct.unpack(f'<{len(data)//2}h', data)
            audio_buffer.extend(chunk)

            while len(audio_buffer) >= FRAME_LENGTH:
                frame = [audio_buffer.popleft() for _ in range(FRAME_LENGTH)]
                keyword_index = porcupine.process(frame)

                if keyword_index >= 0:
                    print("Wake word detected!")
                    await websocket.send_text("DETECTED")

                    # Start dynamic recording
                    post_wake_audio = []
                    silence_threshold = 800  # Amplitude threshold for silence
                    silence_duration_limit = 5.0  # Seconds
                    silence_frame_count = int(SAMPLE_RATE / FRAME_LENGTH * silence_duration_limit)

                    silence_counter = 0
                    print("Recording...")

                    while True:
                        try:
                            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=1.0)
                        except asyncio.TimeoutError:
                            print("Timeout while waiting for audio. Ending recording.")
                            break

                        chunk = struct.unpack(f'<{len(data)//2}h', data)
                        post_wake_audio.extend(chunk)

                        rms = np.sqrt(np.mean(np.square(chunk)))
                        if rms < silence_threshold:
                            silence_counter += 1
                        else:
                            silence_counter = 0  # Reset on speech

                        if silence_counter >= silence_frame_count:
                            print("Silence detected, stopping recording.")
                            break

                    # Transcribe speech
                    transcript = await transcribe_audio(post_wake_audio)
                    print(f"Transcript: {transcript}")
                    await websocket.send_text(f"TRANSCRIPT: {transcript}")
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

async def transcribe_audio(audio_data):
    audio_array = np.array(audio_data, dtype=np.int16)

    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_array.tobytes())
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Google API error: {e}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
