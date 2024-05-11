import os
import re
import unicodedata

# Function to check if the file name contains special characters
def contains_special_characters(file_name):
    pattern = r'[^\x00-\x7F]'
    return re.search(pattern, file_name) is not None

# Function to normalize special characters in the file name
def normalize_file_name(file_name):
    normalized_file_name = unicodedata.normalize('NFD', file_name).encode('ascii', 'ignore').decode('utf-8')
    return normalized_file_name

# Path to the current directory
directory = os.getcwd()

# List of all files in the directory
files = os.listdir(directory)

# Rename files with special characters
for file in files:
    if contains_special_characters(file):
        new_name = normalize_file_name(file)
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
        print(f'File renamed: {file} -> {new_name}')

print("Rename process completed.")