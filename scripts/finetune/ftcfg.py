import os
import glob
import argparse

# Setup argparse to accept var1 and var2 from the command line
parser = argparse.ArgumentParser(description='Modify YAML files.')
parser.add_argument('var1', type=str, help='First variable argument (samples folder).')
parser.add_argument('var2', type=str, help='Second variable argument (finetune destination name in conf/generated/).')
args = parser.parse_args()

var1 = args.var1  # Samples folder
var2 = args.var2  # Finetune destination name in conf/generated/

# Paths of the original YAML files
c2f_file_path = 'conf/finetunetemplates/c2f.yml'
coarse_file_path = 'conf/finetunetemplates/coarse.yml'
interface_file_path = 'conf/finetunetemplates/interface.yml'

# Function to read the content of a YAML file
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

# Read the YAML files
c2f_content = read_yaml_file(c2f_file_path)
coarse_content = read_yaml_file(coarse_file_path)
interface_content = read_yaml_file(interface_file_path)

# Function to update paths in YAML content
def update_paths(yaml_content, new_path, line_number):
    yaml_content[line_number - 1] = f"save_path: {new_path}\n"
    return yaml_content

# Update paths in the YAML files
c2f_content = update_paths(c2f_content, f"./runs/{var2}/c2f", 12)
coarse_content = update_paths(coarse_content, f"./runs/{var2}/coarse", 5)
interface_content[2] = f"Interface.coarse2fine_ckpt: ./runs/{var2}/c2f/latest/vampnet/weights.pth\n"
interface_content[3] = f"Interface.coarse_ckpt: ./runs/{var2}/coarse/latest/vampnet/weights.pth\n"

# Function to count the number of audio files in a folder
def count_audio_files(base_path):
    audio_files = glob.glob(os.path.join(base_path, '*'))
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg'}  # Add more extensions as needed
    return len([file for file in audio_files if os.path.splitext(file)[1].lower() in audio_extensions])

# Count the number of audio files in the specified folder
num_files = count_audio_files(var1)

# Function to generate a list of files with absolute paths
def generate_file_list(base_path):
    audio_files = glob.glob(os.path.join(base_path, '*'))
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg'}  # Add more extensions as needed
    return [os.path.abspath(file) for file in audio_files if os.path.splitext(file)[1].lower() in audio_extensions]

# Generate a list of files based on existing files
file_list = generate_file_list(var1)

# Function to insert a list of files into a YAML file
def insert_file_list_in_yaml(yaml_content, file_list, line_number, is_interface=False):
    formatted_file_list = ["- " + file_path for file_path in file_list]

    if is_interface and formatted_file_list:
        # For the interface file, add the first element on the same line
        yaml_content[line_number - 1] = yaml_content[line_number - 1].rstrip("\n") + formatted_file_list[0] + "\n"
        formatted_file_list = formatted_file_list[1:]

    yaml_content.insert(line_number, "\n".join(formatted_file_list) + "\n")
    return yaml_content

# Insert the list of files into the YAML files
c2f_content = insert_file_list_in_yaml(c2f_content, file_list, 13)
coarse_content = insert_file_list_in_yaml(coarse_content, file_list, 6)
interface_content = insert_file_list_in_yaml(interface_content, file_list, 2, is_interface=True)

# Output directory
output_folder = f'conf/generated/{var2}/'

# Create the directory if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to save modified YAML files
def save_yaml_file(file_path, content):
    with open(file_path, 'w') as file:
        file.writelines(content)

# Save the modified files
save_yaml_file(output_folder + 'c2f.yml', c2f_content)
save_yaml_file(output_folder + 'coarse.yml', coarse_content)
save_yaml_file(output_folder + 'interface.yml', interface_content)

