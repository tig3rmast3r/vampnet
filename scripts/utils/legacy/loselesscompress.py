import os
import sys
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

COMPRESS_EXTENSIONS = {'.wav', '.wave', '.aif', '.aiff'}
EXPAND_EXTENSIONS = {'.flac'}


def convert_audio_to_flac(file_path):
    flac_path = os.path.splitext(file_path)[0] + '.flac'
    audio = AudioSegment.from_file(file_path)
    audio.export(flac_path, format='flac')
    os.remove(file_path)
    print(f"Converted and removed {file_path}")

def convert_flac_to_wav(file_path):
    wav_path = os.path.splitext(file_path)[0] + '.wav'
    audio = AudioSegment.from_file(file_path, format='flac')
    audio.export(wav_path, format='wav')
    os.remove(file_path)
    print(f"Converted and removed {file_path}")


def list_files_recursively(folder_path):
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            yield os.path.join(root, filename)


def process_files(folder_path, operation, num_threads):
    all_files = list(list_files_recursively(folder_path))
    if operation == '-compress':
        files = [
            file_path for file_path in all_files
            if os.path.splitext(file_path)[1].lower() in COMPRESS_EXTENSIONS
        ]
    else:
        files = [
            file_path for file_path in all_files
            if os.path.splitext(file_path)[1].lower() in EXPAND_EXTENSIONS
        ]

    if operation == '-compress':
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(convert_audio_to_flac, files)
    elif operation == '-expand':
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(convert_flac_to_wav, files)

def main():
    if len(sys.argv) < 4:
        print("Usage: python loselesscomp.py -compress|-expand folder_path num_threads")
        sys.exit(1)

    operation = sys.argv[1]
    folder_path = sys.argv[2]
    try:
        num_threads = int(sys.argv[3])
    except ValueError:
        print("Error: num_threads must be an integer.")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    if operation not in ['-compress', '-expand']:
        print("Invalid operation. Use -compress to convert WAV to FLAC or -expand to convert FLAC to WAV.")
        sys.exit(1)

    process_files(folder_path, operation, num_threads)

if __name__ == "__main__":
    main()
