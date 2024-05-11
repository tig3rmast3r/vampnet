import re

def extract_modules_versions(lines):
    module_dict = {}
    for line in lines:
        if not line.startswith('#') and not line.startswith('Package') and not line.startswith('-'):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 2:
                module_name = parts[0].strip().lower()  # Lowercase conversion (for PC)
                module_version = parts[1].strip()
                # Ignore extra columns
                module_dict[module_name] = module_version
    return module_dict

def create_requirements_file(file_path_new, file_path_working, output_file):
    with open(file_path_new, 'r') as file:
        new_data = file.readlines()
    with open(file_path_working, 'r') as file:
        working_data = file.readlines()

    # extract
    new_modules = extract_modules_versions(new_data)
    working_modules = extract_modules_versions(working_data)

    # Create requirements file
    with open(output_file, 'w') as file:
        for module, new_version in new_modules.items():
            if module in working_modules:
                working_version = working_modules[module]
                if new_version != pc_version:
                    file.write(f"{module}=={pc_version}\n")

# paths
file_path_new = 'new'
file_path_working = 'working'
output_file = 'requirements_to_update.txt'

# Run
create_requirements_file(file_path_new, file_path_working, output_file)

