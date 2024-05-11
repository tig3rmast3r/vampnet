from pydub import AudioSegment
import os

def convert_to_mono(file_path):
    audio = AudioSegment.from_wav(file_path)

    # Verifica se il file è già mono
    if audio.channels == 1:
        print(f"Il file {file_path} è già mono.")
        return

    # Converti in mono prendendo solo il canale sinistro
    mono_audio = audio.set_channels(1)

    # Sovrascrivi il file originale con la versione mono
    mono_audio.export(file_path, format="wav")
    print(f"File convertito in mono: {file_path}")

# Percorso della cartella corrente
directory = os.getcwd()

# Elenco di tutti i file WAV nella cartella
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Conversione di ogni file WAV in mono
for wav_file in wav_files:
    convert_to_mono(os.path.join(directory, wav_file))

print("Conversione completata.")