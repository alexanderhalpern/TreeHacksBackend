# %matplotlib qt
import whisperx
import pyaudio
import wave
import tempfile
import os
import requests
import pygame
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
import asyncio
from pyannote.audio import Pipeline
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

load_dotenv()

client = OpenAI()

API_URL = "https://api-inference.huggingface.co/models/lxyuan/distilbert-base-multilingual-cased-sentiments-student"
headers = {"Authorization": "Bearer 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"}
emebeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

if not os.path.exists('embeddings.csv'):
    df = pd.DataFrame(columns=['text', 'embeddings', 'datetime'])
    df.to_csv('embeddings.csv', index=False)

df = pd.read_csv('embeddings.csv')
# Load the whisper model
model = whisperx.load_model("small", "cuda")

# Initialize pygame
pygame.init()

# Function to play sound

conversation_context = [{"role": "system", "content": "You begin your relationship with the user as a stranger who is just meeting the user. You go through common formalities of introductions to the user and you are inquisitive to get to know more and more about the user. As time goes on, you become more friendly and form a close bond with the user. Do not say that you are a virtual assistant or an AI, instead you are a person named Sam."},]


def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

# Function to record audio snippet


async def record_audio(chunk_duration, sample_rate):
    global df
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=int(sample_rate * chunk_duration))

    print("Recording...")

    frames = []
    try:
        while True:
            time = datetime.now()
            data = stream.read(int(sample_rate * chunk_duration))
            frames.append(data)

            # Save the audio snippet to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name
                wave_file = wave.open(temp_filename, 'wb')
                wave_file.setnchannels(1)
                wave_file.setsampwidth(
                    pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wave_file.setframerate(sample_rate)
                wave_file.writeframes(b''.join(frames))
                wave_file.close()

                # Asynchronously transcribe and process the snippet
                await process_snippet(temp_filename)

                # Remove the temporary WAV file
                # Clear frames for the next snippet
                frames = []
            os.remove(temp_filename)
            df.to_csv('embeddings.csv', index=False)
            # print(df)
            print((datetime.now() - time).total_seconds())

    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        # Stop stream and close
        stream.stop_stream()
        stream.close()
        p.terminate()

# Asynchronous function to process the snippet


async def process_snippet(temp_filename):
    global df
    # pipeline = Pipeline.from_pretrained(
    #     "pyannote/speaker-diarization-3.1",
    #     use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

    # # send pipeline to GPU (when available)
    # import torch
    # pipeline.to(torch.device("cuda"))

    # # apply pretrained pipeline
    # diarization = pipeline("audio.wav")

    # print the result
    # for turn, _, speaker in diarization.itertracks(yield_label=True):
    #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    # # start=0.2s stop=1.5s speaker_0
    # start=1.8s stop=3.9s speaker_1
    # start=4.2s stop=5.7s speaker_0
    # ...
    result = model.transcribe(temp_filename)
    if len(result['segments']) > 0:
        transcribedText = result['segments'][0]['text']
        # Replace all Thank you. with ""
        transcribedText = transcribedText.strip().replace("Thank you.", "")
        print("Text:", result)

        if transcribedText == "":
            embeddings = [None]
        else:
            embeddings = emebeddings_model.encode([transcribedText])

        conversation_context.append(
            {"role": "user", "content": transcribedText})

        print(conversation_context)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_context
        )

        gpt_response = response.choices[0].message.content

        url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"

        querystring = {"optimize_streaming_latency": "3"}

        payload = {
            "text": gpt_response,
            "model_id": "eleven_monolingual_v1"
        }
        headers = {
            "xi-api-key": "5a656b58b7ec6201b08eb6bdc60eb256",
            "Content-Type": "application/json"
        }

        response = requests.request(
            "POST", url, json=payload, headers=headers, params=querystring)

        # Check if request was successful
        if response.status_code == 200:
            audio_content = AudioSegment.from_file(BytesIO(response.content))

    # Play the audio
            play(audio_content)
            # Open a file with 'wb' mode (binary mode) to save audio content
            with open("output_audio.mp3", "wb") as file:
                # Write the audio content to the file
                file.write(response.content)
            print("Audio file saved successfully.")
        else:
            print("Failed to retrieve audio:", response.text)

        conversation_context.append({
            "role": "assistant",
            "content": gpt_response
        })

        # Append the embeddings to the dataframe
        df = pd.concat([df, pd.DataFrame([[transcribedText, embeddings, datetime.now()]], columns=[
            'text', 'embeddings', 'datetime'])], ignore_index=True)
        # Perform asynchronous model inferencing
        asyncio.create_task(query_and_process(transcribedText))
# whisper.exr
# Asynchronous function to query the model


async def query_and_process(text):
    response = await asyncio.to_thread(requests.post, API_URL, headers=headers, json={"inputs": text})
    output = response.json()

    # Process the output as needed
    emotionsMap = {}
    for emotion in output[0]:
        emotionsMap[emotion['label']] = emotion['score']

    print(output)

    # Check if the max emotion is 'negative' and play a sound
    if emotionsMap['negative'] > 0.5:
        play_sound('./beep-01a.mp3')

# Set up the initial plot (commented out)
# ...


# Load DF from CSV

# Call the function to start listening to audio and updating the plot
asyncio.run(record_audio(chunk_duration=5, sample_rate=16000))

# Keep the plot open (commented out)
# ...
