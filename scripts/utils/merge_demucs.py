from pydub import AudioSegment
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python mix.py <source_path> <destination_path>")
    sys.exit(1)

source_path = sys.argv[1]
destination_path = sys.argv[2]

tracks = {}

for filename in os.listdir(source_path):
    if filename.endswith(".wav"):
        
        base_name = filename.rsplit('-', 1)[0]

        track_path = os.path.join(source_path, filename)
        audio = AudioSegment.from_wav(track_path)

        if base_name not in tracks:
            tracks[base_name] = audio
        else:
            tracks[base_name] = tracks[base_name].overlay(audio)

for base_name, mixed_audio in tracks.items():
    mixed_audio.export(f"{destination_path}/{base_name}-mixed.wav", format="wav")

print("merge completed")
