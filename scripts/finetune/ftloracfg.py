import os
import glob
import argparse
import yaml

def count_wav_files(base_path):
    """Counts the number of .wav files in a specified path."""
    return len(glob.glob(os.path.join(base_path, '*.wav')))

def calculate_iterations(num_samples, batch_size):
    """Calculates the number of iterations per epoch."""
    return num_samples // batch_size

def update_lora_config_using_lines(file_path, num_samples, iters, sample_freq, val_freq, num_iters):
    """Updates the lora.yml configuration file using line numbers."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Update specific lines
    lines[6] = f"val/AudioDataset.n_examples: {num_samples}\n"  # Line 7
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
parser.add_argument('var1', type=str, help='Finetune sample path in /train/')
parser.add_argument('int1', type=int, help='Epochs validation frequency')
parser.add_argument('int2', type=int, help='Epochs sample(save) frequency')
parser.add_argument('int3', type=int, help='Epochs 1st checkpoint')
parser.add_argument('int4', type=int, help='Epochs 2nd checkpoint')
parser.add_argument('int5', type=int, help='Epochs 3rd checkpoint')
parser.add_argument('int6', type=int, help='Epochs 4th checkpoint')
parser.add_argument('int7', type=int, help='Epochs last checkpoint')
args = parser.parse_args()

# Read and calculate values
num_samples = count_wav_files(os.path.join('train', args.var1))

# Calculate iterations per epoch
iterations_per_epoch = calculate_iterations(num_samples, batch_size)

# Calculate values for save_iters and num_iters
save_iters = [iterations_per_epoch * x for x in [args.int3, args.int4, args.int5, args.int6, args.int7]]
num_iters = iterations_per_epoch * args.int7
sample_freq = iterations_per_epoch * args.int2
val_freq = iterations_per_epoch * args.int1

# Update the lora.yml file
update_lora_config_using_lines(lora_file_path, num_samples, save_iters, sample_freq, val_freq, num_iters)
