from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      model="michellejieli/emotion_text_classifier")


load_dotenv()
client = OpenAI()

pc = Pinecone(api_key='e7a07597-e49c-4873-9d3a-3a679c7bb29d')

pinecone_index = pc.Index(
    host='https://caretaker-vcb4sh5.svc.gcp-starter.pinecone.io',
    name="caretaker", dimension=1536)


def store_clusters_and_embeddings(embeddings, clusters):
    cluster_dict = {}
    for idx, cluster_number in enumerate(clusters):
        if cluster_number not in cluster_dict:
            cluster_dict[cluster_number] = []
        cluster_dict[cluster_number].append(embeddings[idx])
    return cluster_dict


def retrieve_data_from_pinecone():
    # Placeholder for your code to retrieve data from Pinecone
    # For example, you might use Pinecone's Python client
    pinecone_results = pinecone_index.query(
        vector=([0] * 1536), top_k=1000, include_values=True)
    print(len(pinecone_results["matches"]))
    return pinecone_results


def reduce_dimensions(embeddings, n_components=2):
    embeddings_array = np.array([i["values"] for i in embeddings])

    # print(f"Number of samples: {len(embeddings_array)}")

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    # print(reduced_embeddings)
    return reduced_embeddings


def cluster_embeddings(embeddings, n_clusters=4):
    cluster_maps = {}
    for embedding in embeddings:
        results = classifier(embedding["id"])
        label = results[0]["label"]
        print(embedding["id"], label)
        score = results[0]["score"]
        # print(label, score)
        if label not in cluster_maps:
            cluster_maps[label] = [embedding]
        else:
            cluster_maps[label].append(embedding)
    # print(cluster_maps)
    return cluster_maps


def plot_embeddings(embeddings_2d, clusters):
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=clusters, cmap='viridis', marker='o')
    plt.colorbar()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Visualization of Embeddings')
    plt.show()


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


# Assuming you have your embeddings data ready
embeddings = retrieve_data_from_pinecone()["matches"]

if embeddings is not None:
    # Reduce dimensions for visualization
    embeddings_2d = reduce_dimensions(embeddings)

    # Cluster the embeddings
    clusters = cluster_embeddings(embeddings)

    # print(clusters)

    # cluster_embeddings_dict = store_clusters_and_embeddings(
    #     embeddings, clusters)

    # Plot the embeddings in 2D space
    # plot_embeddings(embeddings_2d, clusters)

    # Summarize clusters
    # Note: You will need to implement logic inside summarize_clusters function
    cluster_summaries = summarize_clusters(clusters)

    # Print or process the summaries
    for cluster_number, summary in cluster_summaries.items():
        print(f"{cluster_number} Summary: {summary}")
    # print(clusters["sadness"])
    # print(clusters)
else:
    print("No data retrieved from Pinecone database.")
