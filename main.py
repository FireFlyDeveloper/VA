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
import webrtcvad

load_dotenv()

app = FastAPI()

# Initialize Porcupine wake word engine
porcupine = pvporcupine.create(
    access_key=os.getenv("PORCUPINE_URL"),
    keyword_paths=['hey_ani.ppn']
)

SAMPLE_RATE = porcupine.sample_rate  # likely 16000
FRAME_LENGTH = porcupine.frame_length

# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0-3 (3 = most aggressive, detects smallest speech)

VAD_FRAME_MS = 30
VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples @ 16kHz

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

                    # ---- Start VAD-based recording ----
                    print("Recording (VAD-based)...")

                    post_wake_audio = []
                    vad_buffer = []
                    silence_counter = 0
                    max_silence_frames = 15  # ~0.45s at 30ms frames

                    while True:
                        try:
                            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=1.0)
                        except asyncio.TimeoutError:
                            print("Timeout waiting for audio. Stopping.")
                            break

                        chunk = struct.unpack(f'<{len(data)//2}h', data)
                        post_wake_audio.extend(chunk)
                        vad_buffer.extend(chunk)

                        while len(vad_buffer) >= VAD_FRAME_SIZE:
                            frame_samples = vad_buffer[:VAD_FRAME_SIZE]
                            vad_buffer = vad_buffer[VAD_FRAME_SIZE:]

                            frame_bytes = struct.pack(f"<{VAD_FRAME_SIZE}h", *frame_samples)
                            if vad.is_speech(frame_bytes, SAMPLE_RATE):
                                silence_counter = 0
                            else:
                                silence_counter += 1

                            if silence_counter >= max_silence_frames:
                                print("User stopped speaking. Ending recording.")
                                await websocket.send_text("STOPPED")
                                break
                        else:
                            continue  # continue outer while if inner didn't break
                        break  # break outer while

                    # ---- Transcribe ----
                    transcript = await transcribe_audio(post_wake_audio)

                    if transcript != "Could not understand audio":
                        print(f"Transcript: {transcript}")
                        await websocket.send_text(f"TRANSCRIPT: {transcript}")

                        payload = {"transcription": transcript, "status": "success"}
                        headers = {'Content-Type': 'application/json'}
                        response = requests.post(
                            os.getenv("WEBHOOK_URL"),
                            data=json.dumps(payload),
                            headers=headers
                        )
                        print(f"Webhook response: {response.status_code} - {response.text}")

                        # Generate TTS
                        audio_data = generate_tts(response.text)
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
        tts = gTTS(text=text, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        from pydub import AudioSegment
        sound = AudioSegment.from_mp3(audio_buffer)
        return sound.raw_data
    except Exception as e:
        print(f"TTS generation error: {e}")
        return b''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3300)
