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
from gtts import gTTS
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize Porcupine wake word engine
porcupine = pvporcupine.create(
    access_key=os.getenv("PORCUPINE_URL"),
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
                    silence_threshold = int(os.getenv("silence_threshold"))
                    silence_duration_limit = float(os.getenv("silence_duration_limit"))
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
                            silence_counter = 0

                        if silence_counter >= silence_frame_count:
                            print("Silence detected, stopping recording.")
                            await websocket.send_text("STOPPED")
                            break

                    # Transcribe speech
                    transcript = await transcribe_audio(post_wake_audio)

                    if transcript != "Could not understand audio":
                        print(f"Transcript: {transcript}")
                        await websocket.send_text(f"TRANSCRIPT: {transcript}")
                        
                        payload = {
                            "transcription": transcript,
                            "status": "success"
                        }
                        headers = {'Content-Type': 'application/json'}
                        response = requests.post(
                            os.getenv("WEBHOOK_URL"),
                            data=json.dumps(payload),
                            headers=headers
                        )
                        print(f"\nWebhook response: {response.status_code} - {response.text}")
                        
                        # Generate TTS using gTTS
                        audio_data = generate_tts(response.text)

                        # Send audio data to frontend
                        await websocket.send_bytes(audio_data)

                    else:
                        print("NO_TRANSCRIPT")
                        await websocket.send_text("NO_TRANSCRIPT")

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

def generate_tts(text):
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en')
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert MP3 to WAV
        from pydub import AudioSegment
        sound = AudioSegment.from_mp3(audio_buffer)
        
        # Export as raw PCM data (what your frontend expects)
        raw_data = sound.raw_data
        
        return raw_data
    except Exception as e:
        print(f"TTS generation error: {e}")
        return b''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)