import torch
import sys

# Soglia per determinare se un gruppo è troppo grande per essere visualizzato completamente
MAX_GROUP_SIZE = 10 * 1024  # 10 KB

def display_keys(data):
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key} (group)")
        else:
            size = sys.getsizeof(value)
            if size <= MAX_GROUP_SIZE:
                print(f"{key}: {value}")
            else:
                print(f"{key}: too big")

def display_group(data):
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key} (group)")
        else:
            print(f"{key}: {value}")

def get_nested_value(data, keys):
    for key in keys:
        data = data[key]
    return data

def set_nested_value(data, keys, value):
    for key in keys[:-1]:
        data = data[key]
    data[keys[-1]] = value

def modify_value(data, keys):
    current_value = get_nested_value(data, keys)
    print(f"Current value of {keys[-1]}: {current_value}")
    new_value = input(f"Enter new value for {keys[-1]}: ")
    try:
        new_value = eval(new_value)  # Safely convert input to appropriate type
    except:
        print("Invalid input. Modification aborted.")
        return False
    set_nested_value(data, keys, new_value)
    print(f"Value of {keys[-1]} updated to: {new_value}")
    return True

def modify_key_name(data, keys):
    key_to_modify = keys[-1]
    parent_data = get_nested_value(data, keys[:-1])
    new_key_name = input(f"Enter new name for the key '{key_to_modify}': ")
    if new_key_name in parent_data:
        print(f"Key '{new_key_name}' already exists. Modification aborted.")
        return False
    parent_data[new_key_name] = parent_data.pop(key_to_modify)
    print(f"Key name '{key_to_modify}' changed to '{new_key_name}'")
    return True

def add_key(data, keys):
    parent_data = get_nested_value(data, keys)
    new_key_name = input("Enter the new key name: ")
    if new_key_name in parent_data:
        print(f"Key '{new_key_name}' already exists. Addition aborted.")
        return False
    new_value = input(f"Enter the value for the new key '{new_key_name}': ")
    try:
        new_value = eval(new_value)  # Safely convert input to appropriate type
    except:
        print("Invalid input. Addition aborted.")
        return False
    parent_data[new_key_name] = new_value
    print(f"Key '{new_key_name}' with value '{new_value}' added.")
    return True

def navigate(data, path=[]):
    current_data = get_nested_value(data, path) if path else data
    display_keys(current_data)
    
    while True:
        command = input("Enter the key you want to view/modify/add, 'back' to go up, or 'exit' to quit: ")
        if command == 'exit':
            return False
        elif command == 'back':
            if path:
                path.pop()
                current_data = get_nested_value(data, path) if path else data
                display_keys(current_data)
            else:
                print("Already at the top level.")
        elif command in current_data:
            selected_value = current_data[command]
            if isinstance(selected_value, dict):
                path.append(command)
                current_data = selected_value
                display_keys(current_data)
            else:
                print(f"{command}: {selected_value}")
                action = input("Do you want to modify the value, the key name, or add a new key? (value/key/add): ").lower()
                if action == 'value':
                    if modify_value(data, path + [command]):
                        return True
                elif action == 'key':
                    if modify_key_name(data, path + [command]):
                        return True
                elif action == 'add':
                    if add_key(data, path + [command]):
                        return True
                else:
                    print("Invalid action. Please try again.")
        elif command == 'add':
            if add_key(data, path):
                return True
        else:
            print("Invalid key. Please try again.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python pth_reader.py <file_path>")
        return

    file_path = sys.argv[1]
    data = torch.load(file_path)
    modifications = []

    print("Keys:")
    while True:
        if navigate(data):
            modifications.append(True)
        else:
            break

    if modifications:
        if input("Do you want to save the changes? (y/n): ").lower() == 'y':
            torch.save(data, file_path)
            print("Changes saved.")
        else:
            print("Changes discarded.")

if __name__ == "__main__":
    main()

