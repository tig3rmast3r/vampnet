import os
import subprocess

# Convert M4A to WAV.
def convert_to_wav(m4a_file, wav_file):
    command = f'ffmpeg -i "{m4a_file}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{wav_file}"'
    subprocess.call(command, shell=True)

# Current working directory.
directory = os.getcwd()

# List all M4A files in the directory.
m4a_files = [f for f in os.listdir(directory) if f.endswith('.m4a')]

# Convert each M4A file to WAV.
for m4a_file in m4a_files:
    wav_file = m4a_file.replace('.m4a', '.wav')
    convert_to_wav(m4a_file, wav_file)
    print(f'Converted: {m4a_file} -> {wav_file}')

print("Conversion complete.")
