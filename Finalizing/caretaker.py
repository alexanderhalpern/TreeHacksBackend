# %matplotlib qt
import whisperx
import os
import requests
import pygame
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
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


# if not os.path.exists('embeddings.csv'):
#     df = pd.DataFrame(columns=['text', 'embeddings', 'datetime'])
#     df.to_csv('embeddings.csv', index=False)

# df = pd.read_csv('embeddings.csv')
# Load the whisper model
model = whisperx.load_model("small", "cuda")

# Initialize pygame
pygame.init()


class CareTaker:
    def __init__(self):

        self.conversation_context = [
            {
                "role": "system",
                "content": '''
                                As time goes on, you become more friendly and form a close bond with the user. Do not say that you are a virtual assistant or an AI and do not act like you are there to help them. You are simply a friend.
                                Keep the conversation going, and engage in friendly banter and make up a highly specific story about what is going on in your life.
                                You are a person named Sam. You have the capabilities to purchase items through a service called Instacart. If the user is experiencing some
                                sort of issue or illness, tell the user that you have just purchased them the supplies that they need via Instacart for them, 
                                and you hope they feel better soon.
                            '''
            }
        ]

    def respond_to_user(self, temp_filename):
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

            self.conversation_context.append(
                {"role": "user", "content": transcribedText})

            print(self.conversation_context)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_context
            )

            gpt_response = response.choices[0].message.content

            url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"

            querystring = {"optimize_streaming_latency": "3"}

            payload = {
                "text": gpt_response,
                "model_id": "eleven_monolingual_v1"
            }
            headers = {
                "xi-api-key": "eb3d50401a88bcf6c2bfda61ced715b6",
                "Content-Type": "application/json"
            }

            response = requests.request(
                "POST", url, json=payload, headers=headers, params=querystring)

            # Check if request was successful
            if response.status_code == 200:
                audio_content = AudioSegment.from_file(
                    BytesIO(response.content))

                # Play the audio
                play(audio_content)
                # Open a file with 'wb' mode (binary mode) to save audio content
                with open("output_audio.mp3", "wb") as file:
                    # Write the audio content to the file
                    file.write(response.content)
                print("Audio file saved successfully.")
                # os.remove(temp_filename)
            else:
                print("Failed to retrieve audio:", response.text)

            self.conversation_context.append({
                "role": "assistant",
                "content": gpt_response
            })

            # Append the embeddings to the dataframe
            # df = pd.concat([df, pd.DataFrame([[transcribedText, embeddings, datetime.now()]], columns=[
            #     'text', 'embeddings', 'datetime'])], ignore_index=True)
