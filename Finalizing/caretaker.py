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
from pinecone import Pinecone
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline


pc = Pinecone(api_key='e7a07597-e49c-4873-9d3a-3a679c7bb29d')

load_dotenv()

client = OpenAI()

API_URL = "https://api-inference.huggingface.co/models/lxyuan/distilbert-base-multilingual-cased-sentiments-student"
headers = {"Authorization": "Bearer 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"}
emebeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

classifier = pipeline("sentiment-analysis",
                      model="michellejieli/emotion_text_classifier")


# if not os.path.exists('embeddings.csv'):
#     df = pd.DataFrame(columns=['text', 'embeddings', 'datetime'])
#     df.to_csv('embeddings.csv', index=False)

# df = pd.read_csv('embeddings.csv')
# Load the whisper model
model = whisperx.load_model("small", "cuda")

# Initialize pygame
pygame.init()


def generateEmbeddings(text):
    text = text.replace("\n", " ").strip()
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding


class CareTaker:
    def __init__(self):
        self.pinecone_index = pc.Index(
            host='https://caretaker-vcb4sh5.svc.gcp-starter.pinecone.io',
            name="caretaker", dimension=1536)
        self.conversation_context = [
            {
                "role": "system",
                "content": '''
                            As time goes on, you become more friendly and form a close bond with the user. Do not say that you are a virtual assistant or an AI and do not act like you are there to help them. You are simply a friend.
                            Keep the conversation going, and engage in friendly banter and SOMETIMES make up a highly specific story about what is going on in your life.
                            You are a person named Sam. You have the capabilities to purchase items through a service called Instacart. Only if the user is experiencing some
                            sort of issue or illness, tell the user that you have just purchased them the supplies that they need via Instacart for them, 
                            and you hope they feel better soon. Don't provide any specifics as to what you bought them.
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

            initial_memories = self.pinecone_index.query(
                vector=generateEmbeddings(transcribedText), top_k=3, include_metadata=True)

            memories = []
            for im in initial_memories["matches"]:
                if "label" not in im["metadata"] or im["metadata"]["label"] == "joy" or im["metadata"]["label"] == "neutral":
                    memories.append(im)

            print(memories)

            if len(memories) > 2:
                user_request = f'''
                Here is a record of a past message that the user sent you that might be related to the user's next message. You can choose to use or ignore this information depending on what you see fit.
                {memories[0]['id']}
                {memories[1]['id']}
                {memories[2]['id']}
                '''
                print(user_request)

                self.conversation_context.append(
                    {"role": "system", "content": user_request})

            self.conversation_context.append(
                {"role": "user", "content": transcribedText})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_context
            )

            gpt_response = response.choices[0].message.content
            print(gpt_response)

            if "Instacart" in gpt_response:
                print("WE ORDERING")

            self.conversation_context.append({
                "role": "assistant",
                "content": gpt_response
            })

            # print(gpt_response)

            # url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"

            # querystring = {"optimize_streaming_latency": "3"}

            # payload = {
            #     "text": gpt_response,
            #     "model_id": "eleven_monolingual_v1"
            # }
            # headers = {
            #     "xi-api-key": "eb3d50401a88bcf6c2bfda61ced715b6",
            #     "Content-Type": "application/json"
            # }

            # response = requests.request(
            #     "POST", url, json=payload, headers=headers, params=querystring)

            # # Check if request was successful
            # if response.status_code == 200:
            #     audio_content = AudioSegment.from_file(
            #         BytesIO(response.content))

            #     # Play the audio
            #     play(audio_content)
            #     # Open a file with 'wb' mode (binary mode) to save audio content
            #     with open("output_audio.mp3", "wb") as file:
            #         # Write the audio content to the file
            #         file.write(response.content)
            #     print("Audio file saved successfully.")
            #     # os.remove(temp_filename)
            # else:
            #     print("Failed to retrieve audio:", response.text)

            # self.conversation_context.append({
            #     "role": "assistant",
            #     "content": gpt_response
            # })

            if transcribedText == "":
                embeddings = [None]
            else:
                embeddings = generateEmbeddings(transcribedText)

            results = classifier(transcribedText)
            label = results[0]["label"]

            self.pinecone_index.upsert(vectors=[{
                "id": re.compile(r'[^\x00-\x7F]+').sub('', transcribedText),
                "values": embeddings,
                "metadata": {"date": str(datetime.now()), "label": label}
            }])

            return self.conversation_context

            # Append the embeddings to the dataframe
            # df = pd.concat([df, pd.DataFrame([[transcribedText, embeddings, datetime.now()]], columns=[
            #     'text', 'embeddings', 'datetime'])], ignore_index=True)
