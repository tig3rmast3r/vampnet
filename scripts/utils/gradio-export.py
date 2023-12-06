#simple utility to extract output.wav from gradio-outputs, wav files will be saved into gradio-export folder, timestamp will be added ad prefix

import os
import shutil
from datetime import datetime

# abs path
script_dir = os.path.dirname(os.path.realpath(__file__))

# base path
base_dir = os.path.dirname(os.path.dirname(script_dir))

# gradio path
output_dir = os.path.join(base_dir, "gradio-outputs")
#input_dir = os.path.join(base_dir, "gradio-outputs/tmp")

# export path
destination_dir = os.path.join(base_dir, "gradio-export")

# create folder
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# copy and rename method
def copy_and_rename_files(source_dir, file_name, file_suffix):
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y%m%d%H%M%S')
                new_file_name = f"{timestamp}_{file_suffix}.wav"
                new_file_path = os.path.join(destination_dir, new_file_name)
                shutil.copy2(file_path, new_file_path)


copy_and_rename_files(output_dir, 'output.wav', 'output')
#copy_and_rename_files(input_dir, 'input.wav', 'input')

print("completed")