import pyaudio
import wave
import tempfile
import audioop
from caretaker import CareTaker


class Elder:
    def __init__(self):
        self.silence_detected = False
        self.speech_volume = 100
        self.careTaker = CareTaker()

    def listen(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        wait_time = 20

        audio = pyaudio.PyAudio()

        # Open stream
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            # input_device_index=2,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        print("Recording...")

        frames = []
        recording = False
        frames_recorded = 0

        while True:
            frames_recorded += 1
            data = stream.read(CHUNK)
            rms = audioop.rms(data, 2)

            if not self.silence_detected:
                if frames_recorded < 40:
                    if frames_recorded == 1:
                        print("Detecting ambient noise...")
                    if frames_recorded > 5:
                        if self.speech_volume < rms:
                            self.speech_volume = rms
                        print("Speech volume threshold:", self.speech_volume)
                    continue
                elif frames_recorded == 40:
                    print("Listening...")
                    # Adjust threshold based on ambient noise
                    self.speech_volume = self.speech_volume * 3
                    self.silence_detected = True

            if rms > self.speech_volume:
                if not recording:
                    print("Start Recording...")
                    frames = []  # Start a new recording
                recording = True
                frames_recorded = 0
            elif recording and frames_recorded > wait_time:
                print("Stop Recording...")
                recording = False

                # Save the audio snippet to a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_filename = temp_wav.name
                    wave_file = wave.open(temp_filename, 'wb')
                    wave_file.setnchannels(1)
                    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
                    wave_file.setframerate(RATE)
                    wave_file.writeframes(b''.join(frames))
                    wave_file.close()
                    # Asynchronously transcribe and process the snippet
                    self.careTaker.respond_to_user(temp_filename)

                # Remove the temporary WAV file
                # os.remove(temp_filename)

                # Clear frames for the next snippet
                frames = []

                # df.to_csv('embeddings.csv', index=False)

            if recording:
                frames.append(data)
