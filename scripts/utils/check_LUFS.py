import os
import sys
import soundfile as sf
import pyloudnorm as pyln

def check_normalization(folder_path):
    if not os.path.exists(folder_path):
        print(f'Error: "{folder_path}" not found.')
        return

    for file in os.listdir(folder_path):
        if file.endswith(('.wav', '.mp3')):
            file_path = os.path.join(folder_path, file)
            try:
                data, rate = sf.read(file_path)
                meter = pyln.Meter(rate)
                loudness = meter.integrated_loudness(data)
                print(f'"{file}" LUFS = {loudness:.2f}')
            except Exception as e:
                print(f'Errore reading file "{file}": {str(e)}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    check_normalization(folder_path)
