from pydub import AudioSegment
import os

def convert_to_mono(file_path):
    audio = AudioSegment.from_wav(file_path)

    # Skip files that are already mono.
    if audio.channels == 1:
        print(f"File is already mono: {file_path}")
        return

    # Convert to mono by selecting one channel.
    mono_audio = audio.set_channels(1)

    # Overwrite the original file with the mono version.
    mono_audio.export(file_path, format="wav")
    print(f"Converted to mono: {file_path}")

# Current working directory.
directory = os.getcwd()

# List all WAV files in the directory.
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Convert each WAV file to mono.
for wav_file in wav_files:
    convert_to_mono(os.path.join(directory, wav_file))

print("Conversion complete.")
