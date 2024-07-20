import os
import glob
import argparse
import yaml
from math import gcd
from functools import reduce

def count_audio_files(base_path):
    """Counts the number of audio files in a specified path."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg'}  # Add more extensions as needed
    audio_files = glob.glob(os.path.join(base_path, '*'))
    return len([file for file in audio_files if os.path.splitext(file)[1].lower() in audio_extensions])

def calculate_iterations(num_samples, batch_size):
    """Calculates the number of iterations per epoch."""
    return num_samples // batch_size

def lcm(a, b):
    """Calculates the least common multiple of two integers."""
    return abs(a * b) // gcd(a, b)

def lcmm(*args):
    """Calculates the least common multiple of multiple integers."""
    return reduce(lcm, args)

def round_to_nearest_multiple(number, multiple):
    """Rounds a number to the nearest multiple of another number."""
    return multiple * round(number / multiple)

def update_lora_config_using_lines(file_path, num_val, iters, sample_freq, val_freq, num_iters):
    """Updates the lora.yml configuration file using line numbers."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Update specific lines
    lines[6] = f"val/AudioDataset.n_examples: {num_val}\n"  # Line 7
    lines[13] = f"save_iters: {iters}\n"  # Line 14
    lines[14] = f"sample_freq: {sample_freq}\n"  # Line 15
    lines[15] = f"val_freq: {val_freq}\n"  # Line 16
    lines[21] = f"num_iters: {num_iters}\n"  # Line 22
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Read the batch size from lora.yml
lora_file_path = 'conf/lora/lora.yml'
with open(lora_file_path, 'r') as file:
    lora_config = yaml.safe_load(file)
batch_size = lora_config['batch_size']

# Setup argparse
parser = argparse.ArgumentParser(description='Configure lora.yml.')
parser.add_argument('var1', type=str, help='Finetune sample path')
parser.add_argument('var2', type=str, help='Validation sample path')
parser.add_argument('int1', type=float, help='Epochs validation frequency (can be fractional)')
parser.add_argument('int2', type=float, help='Epochs sample (save) frequency (can be fractional)')
parser.add_argument('int3', type=int, help='Epochs 1st checkpoint')
parser.add_argument('int4', type=int, help='Epochs 2nd checkpoint')
parser.add_argument('int5', type=int, help='Epochs 3rd checkpoint')
parser.add_argument('int6', type=int, help='Epochs 4th checkpoint')
parser.add_argument('int7', type=int, help='Epochs last checkpoint')
args = parser.parse_args()

# Read and calculate values
num_samples = count_audio_files(args.var1)
num_val = count_audio_files(args.var2)

# Calculate iterations per epoch
iterations_per_epoch = calculate_iterations(num_samples, batch_size)

# Calculate values for save_iters and num_iters
sample_freq = int(round(args.int2 * iterations_per_epoch))
val_freq = int(round(args.int1 * iterations_per_epoch))
save_iters = [int(round_to_nearest_multiple(iterations_per_epoch * x, sample_freq)) for x in [args.int3, args.int4, args.int5, args.int6, args.int7]]
num_iters = save_iters[-1]

# Update the lora.yml file
update_lora_config_using_lines(lora_file_path, num_val, save_iters, sample_freq, val_freq, num_iters)

