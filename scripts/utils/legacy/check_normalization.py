import os
import sys
import soundfile as sf
import numpy as np

def peak_normalize_audio(file_path, data, rate):
    peak = np.max(np.abs(data))
    normalized_data = data / peak * 0.988  # Normalize -0.1db
    sf.write(file_path, normalized_data, rate)
    print(f'Normalized: {file_path}')

def check_peak_normalization(folder_path):
    files_to_normalize = []

    for file in os.listdir(folder_path):
        if file.endswith(('.wav', '.mp3')):
            file_path = os.path.join(folder_path, file)
            try:
                data, rate = sf.read(file_path)
                peak = np.max(np.abs(data))
                if peak < 0.987:
                    print(f'File "{file}" is not normalized. Peak: {20 * np.log10(peak):.2f} dBFS')
                    files_to_normalize.append((file_path, data, rate))
            except Exception as e:
                print(f'Error reading file "{file}": {str(e)}')

    if files_to_normalize:
        response = input("Do you want to normalize files? (y/n): ")
        if response.lower() == 'y':
            for file_path, data, rate in files_to_normalize:
                peak_normalize_audio(file_path, data, rate)
        else:
            print("Nothing changed")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    check_peak_normalization(folder_path)
