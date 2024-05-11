import os
import subprocess

# Funzione per convertire M4A in WAV
def convert_to_wav(m4a_file, wav_file):
    command = f'ffmpeg -i "{m4a_file}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{wav_file}"'
    subprocess.call(command, shell=True)

# Percorso della cartella corrente
directory = os.getcwd()

# Elenco di tutti i file M4A nella cartella
m4a_files = [f for f in os.listdir(directory) if f.endswith('.m4a')]

# Conversione di ogni file M4A in WAV
for m4a_file in m4a_files:
    wav_file = m4a_file.replace('.m4a', '.wav')
    convert_to_wav(m4a_file, wav_file)
    print(f'Convertito: {m4a_file} in {wav_file}')

print("Conversione completata.")