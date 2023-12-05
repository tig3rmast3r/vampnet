#universal chunk creator - input must be MP3 or WAV
#usage (from vampnet folder) python scripts/utils/ceatechunks.py "<absolute sourcedir>" <destination folder>
#destination folder will be saved into vampnet/train/
#it will search recursively if there are additional folders inside the main one
#Case 1 - audio is less than 10 seconds, i assume is a loop so it will be repeated as many time as needed to reach 10 seconds
#Case 2 - audio is from 10 to 69 sec. May be a manual recording or a longer loop, it will trim starting blank(if any) and trim exceeding audio
#Case 3 - file is longer than 69 sec, may be a track or a mixed compilation, it will save a chunk every 10 seconds, starting from 60 seconds
#this is intended for electronic music that is usually very repetitive and splitting audio every 10 sec will bring lot of redundancy
#final check - if chunk is not blank or too low it will be normalized and then saved, will also fix to 16 bit 44100hz
import os
import sys
from pydub import AudioSegment, silence

def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=10):
    """
    Detects the leading silence in an audio segment.
    """
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def is_audio_not_silent(audio_segment):
    """
    Checks if the audio segment is not silent.
    """
    return audio_segment.dBFS > -30.0

def process_audio(file_path, output_dir, file_name):
    # Carica il file audio in base al suo formato
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
    elif file_path.endswith('.wav'):
        audio = AudioSegment.from_wav(file_path)
    else:
        return []  # Salta i file che non sono mp3 o wav

    # Conversione a mono e impostazione del sample rate
    audio = audio.set_channels(1).set_frame_rate(44100)

    duration_ms = len(audio)
    output_files = []

    # Case 1
    if duration_ms < 10000:
        repeats = 10000 // duration_ms + 1
        audio = audio * repeats
        audio = audio[:10000]  # Trim to exactly 10 seconds
        if is_audio_not_silent(audio):
            audio = audio.normalize(headroom=0.1)
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wav")
            audio.export(output_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
            output_files.append(output_file)

    # Case 2
    elif duration_ms <= 69000:
        start_trim = detect_leading_silence(audio, silence_threshold=-30.0)
        end_trim = start_trim + 10000
        if end_trim <= duration_ms:
            audio = audio[start_trim:end_trim]
            if is_audio_not_silent(audio):
                audio = audio.normalize(headroom=0.1)
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wav")
                audio.export(output_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
                output_files.append(output_file)
    #Case 3
    else:
        for i in range(1, duration_ms // 60000 + 1):
            start_ms = i * 60000
            end_ms = start_ms + 10000
            if end_ms > duration_ms:
                break
            chunk = audio[start_ms:end_ms]

            if is_audio_not_silent(chunk):
                chunk = chunk.normalize(headroom=0.1)
                chunk_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{str(i).zfill(3)}.wav")
                chunk.export(chunk_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
                output_files.append(chunk_file)

    return output_files

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_directory> <output_subdirectory>")
        sys.exit(1)

    source_dir = sys.argv[1]
    output_subdir = sys.argv[2]
    output_dir = os.path.join('train', output_subdir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                print(f"Processing {file}...")
                process_audio(file_path, output_dir, file)

if __name__ == "__main__":
    main()
