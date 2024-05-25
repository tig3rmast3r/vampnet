#universal chunk creator - input must be MP3,WAV,AAC,M4A
#usage (from vampnet folder) python scripts/utils/createchunks.py "<absolute sourcedir>" <destination folder> [-log] [-verbose]
#destination folder will be saved into vampnet/train/
#it will search recursively if there are additional folders inside the main one
#Case 1 - audio is less than 10 seconds, i assume is a loop so it will be repeated as many time as needed to reach 10 seconds
#Case 1b - audio is exactly 10 seconds, will check for leading and trailing silence and if is blank. If there's still audio after trim will jump to Case 1
#Case 2 - audio is from 10 to 70 sec. May be a manual recording or a longer loop, it will trim leading blank and trim exceeding audio
#Case 3 - file is longer than 70 sec, may be a track or a mixed compilation, it will save a chunk every 60 seconds, starting from 60 seconds
#this is intended for electronic music that is usually very repetitive and splitting audio every 10 sec will bring lot of redundancy
#final check - if chunk is not blank or too low it will be normalized and then saved, will also fix to 16 bit 44100hz
import os
import sys
import logging
from pydub import AudioSegment, silence

def configure_logging(log_to_file, verbose):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to the lowest level to capture all messages

    if log_to_file:
        file_handler = logging.FileHandler('audio_processing.log')
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    """
    Detects the leading silence in an audio segment.
    """
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms

def detect_trailing_silence(sound, silence_threshold=-40.0, chunk_size=10):
    """
    Detects the trailing silence in an audio segment.
    """
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while trim_ms < len(sound) and sound[-trim_ms-chunk_size:-trim_ms].dBFS < silence_threshold:
        trim_ms += chunk_size

    if trim_ms >= len(sound):
        trim_ms = len(sound)
    
    return trim_ms

def is_audio_not_silent(audio_segment):
    """
    Checks if the audio segment is not silent.
    """
    return audio_segment.dBFS > -40.0

def process_audio(file_path, output_dir, file_name):
    # load file
    try:
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
        elif file_path.endswith('.wav'):
            audio = AudioSegment.from_wav(file_path)
        elif file_path.endswith('.m4a'):
            audio = AudioSegment.from_file(file_path, format='m4a')
        elif file_path.endswith('.aac'):
            audio = AudioSegment.from_file(file_path, format='aac')
        else:
            logging.warning(f"File {file_path} skipped: unsupported format.")
            return []  # skip if not audio
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []  # skip if error

    # fix channels and sample rate
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
            if not os.path.exists(output_file):
                audio.export(output_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
                output_files.append(output_file)
            else:
                logging.warning(f"File {output_file} already exists. Skipping.")
        else:
            logging.warning(f"File {file_path} skipped: silence detected in the audio.")

    # Case 1b
    elif duration_ms == 10000:
        start_trim = detect_leading_silence(audio, silence_threshold=-40.0)
        if start_trim >= duration_ms:  # Entire audio is silent
            logging.warning(f"File {file_path} skipped: entire audio is silent.")
            return output_files
        
        if start_trim > 1000:
            audio = audio[start_trim:]
        
        end_trim = detect_trailing_silence(audio, silence_threshold=-40.0)
        if end_trim > 1000:
            audio = audio[:-end_trim]

        if len(audio) > 1000:  # Check if the remaining audio is greater than 1 second
            repeats = 10000 // len(audio) + 1
            audio = audio * repeats
            audio = audio[:10000]  # Trim to exactly 10 seconds
            if is_audio_not_silent(audio):
                audio = audio.normalize(headroom=0.1)
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wav")
                if not os.path.exists(output_file):
                    audio.export(output_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
                    output_files.append(output_file)
                else:
                    logging.warning(f"File {output_file} already exists. Skipping.")
            else:
                logging.warning(f"File {file_path} skipped: silence detected in the audio.")
        else:
            logging.warning(f"File {file_path} skipped: remaining audio less than 1 second after trimming silence.")
    
    # Case 2
    elif duration_ms <= 69999:
        start_trim = detect_leading_silence(audio, silence_threshold=-40.0)
        end_trim = start_trim + 10000
        if end_trim <= duration_ms:
            audio = audio[start_trim:end_trim]
            if is_audio_not_silent(audio):
                audio = audio.normalize(headroom=0.1)
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wav")
                if not os.path.exists(output_file):
                    audio.export(output_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
                    output_files.append(output_file)
                else:
                    logging.warning(f"File {output_file} already exists. Skipping.")
            else:
                logging.warning(f"File {file_path} skipped: silence detected in the audio.")
        else:
            logging.warning(f"File {file_path} skipped: not enough duration after trimming silence.")
    # Case 3
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
                if not os.path.exists(chunk_file):
                    chunk.export(chunk_file, format='wav', parameters=["-ac", "1", "-ar", "44100", "-sample_fmt", "s16"])
                    output_files.append(chunk_file)
                else:
                    logging.warning(f"File {chunk_file} already exists. Skipping.")
            else:
                logging.warning(f"Chunk {i} of file {file_path} skipped: silence detected in the audio.")
    return output_files

def main():
    if len(sys.argv) < 3:
        print("Usage: python createchunks.py <source_directory> <output_subdirectory> [-log] [-verbose]")
        sys.exit(1)

    source_dir = sys.argv[1]
    output_subdir = sys.argv[2]
    log_to_file = '-log' in sys.argv
    verbose = '-verbose' in sys.argv

    configure_logging(log_to_file, verbose)

    output_dir = os.path.join('train', output_subdir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.m4a', '.aac')):
                file_path = os.path.join(root, file)
                output_file_base = os.path.join(output_dir, f"{os.path.splitext(file)[0]}")
                if any(os.path.exists(f"{output_file_base}_{str(i).zfill(3)}.wav") for i in range(1, (len(files) // 60000) + 2)):
                    logging.warning(f"Skipping {file} as it already exists in the output directory.")
                    continue
                logging.info(f"Processing {file}...")
                process_audio(file_path, output_dir, file)

if __name__ == "__main__":
    main()