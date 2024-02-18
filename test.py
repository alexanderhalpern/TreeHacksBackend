import requests

url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"

querystring = {"optimize_streaming_latency": "3"}

payload = {
    "text": "Hi Margaret, its nice to meet you.",
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
    # Open a file with 'wb' mode (binary mode) to save audio content
    with open("output_audio.mp3", "wb") as file:
        # Write the audio content to the file
        file.write(response.content)
    print("Audio file saved successfully.")
else:
    print("Failed to retrieve audio:", response.text)
