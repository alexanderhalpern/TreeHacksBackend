from flask import Flask, request
from caretaker import CareTaker
import tempfile
# import audioop
# import pyaudio
# import os
from flask_cors import CORS
# from pydub import AudioSegment
# from pydub.playback import play
from openai import OpenAI
from transformers import pipeline
from pinecone import Pinecone


classifier = pipeline("sentiment-analysis",
                      model="michellejieli/emotion_text_classifier")


app = Flask(__name__)
CORS(app)
client = OpenAI()

pc = Pinecone(api_key='e7a07597-e49c-4873-9d3a-3a679c7bb29d')

pinecone_index = pc.Index(
    host='https://caretaker-vcb4sh5.svc.gcp-starter.pinecone.io',
    name="caretaker", dimension=1536)


def cluster_embeddings(embeddings, n_clusters=4):
    cluster_maps = {}
    for embedding in embeddings:
        results = classifier(embedding["id"])
        label = results[0]["label"]
        # print(label, score)
        if label not in cluster_maps:
            cluster_maps[label] = [embedding]
        else:
            cluster_maps[label].append(embedding)
    # print(cluster_maps)
    return cluster_maps


def retrieve_data_from_pinecone():
    # Placeholder for your code to retrieve data from Pinecone
    # For example, you might use Pinecone's Python client
    pinecone_results = pinecone_index.query(
        vector=([0] * 1536), top_k=1000, include_values=True)
    print(len(pinecone_results["matches"]))
    return pinecone_results


def summarize_clusters(clusters):
    cluster_summaries = {}
    for cluster_name, embeddings in clusters.items():
        # Assuming each embedding has an associated ID, concatenate them
        # print(embeddings)
        # print(cluster_name)
        # print(embeddings)
        cluster_ids = " ".join([str(embedding["id"])
                               for embedding in embeddings])

        # print(cluster_ids)
        # Create a summary prompt for this cluster
        summary_prompt = f"In one sentence summarize the most important topics of these items. Do not address anything besides these topics: {cluster_ids}"

        # Use OpenAI's API to get the summary (this is a placeholder)
        # Replace the below line with your API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": summary_prompt}]
        )

        # Store the summary in the dictionary
        cluster_summaries[cluster_name] = response.choices[0].message.content.strip(
        )

    return cluster_summaries


@app.route('/sentimentAnalysis', methods=['POST'])
def sentimentAnalysis():
    # Assuming you have your embeddings data ready
    embeddings = retrieve_data_from_pinecone()["matches"]
    if embeddings is not None:
        clusters = cluster_embeddings(embeddings)
        sentimentSum = 0
        numberClusters = 0
        if "joy" in clusters:
            sentimentSum += 1 * (len(clusters["joy"]))
            numberClusters += (len(clusters["joy"]))
        if "sadness" in clusters:
            numberClusters += (len(clusters["sadness"]))
        if "neutral" in clusters:
            sentimentSum += 0.5 * (len(clusters["joy"]))
            numberClusters += (len(clusters["joy"]))

        cluster_summaries = summarize_clusters(clusters)
        return {
            "cluster_summaries": cluster_summaries,
            "sentiment_score": sentimentSum / numberClusters
        }
    return {}


@app.route("/receiveAudio", methods=['POST'])
def receiveAudio():
    # Check if the post request has the file part
    careTaker = CareTaker()

    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']

    print(file)

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:

            temp_filename = temp_file.name
            print(temp_filename)
            file.save(temp_filename)
            print(temp_filename)
            file.save("audio.wav")

            # Play the audio
            # audio = AudioSegment.from_file(
            #     temp_filename, format="mp3")  # Adjust format as needed
            # play(audio)
            return careTaker.respond_to_user(temp_filename), 200
    return 500

    # Process the file or return response

    # file = request.files['file']
    # FORMAT = pyaudio.paInt16
    # CHANNELS = 1
    # RATE = 44100
    # CHUNK = 1024
    # wait_time = 20
    # careTaker = CareTaker()
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
    #     temp_filename = temp_wav.name
    #     wave_file = wave.open(temp_filename, 'wb')
    #     wave_file.setnchannels(1)
    #     wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    #     wave_file.setframerate(RATE)
    #     wave_file.writeframes(b''.join(frames))
    #     wave_file.close()
    #     # Asynchronously transcribe and process the snippet
    #     self.careTaker.respond_to_user(temp_filename)
if __name__ == "__main__":
    app.run(debug=True)
