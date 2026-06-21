import sys
import os
import torch

def update_files(folder_path, batch_size, noam_warmup, noam_factor, step, lr):
    tracker_path = os.path.join(folder_path, 'tracker.pth')
    scheduler_path = os.path.join(folder_path, 'scheduler.pth')
    optimizer_path = os.path.join(folder_path, 'optimizer.pth')

    # Update tracker.pth file
    if not os.path.exists(tracker_path):
        print(f"Error: {tracker_path} not found.")
        return
    tracker_data = torch.load(tracker_path)
    tracker_data['step'] = step
    torch.save(tracker_data, tracker_path)

    # Update scheduler.pth file
    if not os.path.exists(scheduler_path):
        print(f"Error: {scheduler_path} not found.")
        return
    scheduler_data = torch.load(scheduler_path)
    scheduler_data['warmup'] = noam_warmup
    scheduler_data['factor'] = noam_factor
    scheduler_data['steps'] = step
    torch.save(scheduler_data, scheduler_path)

    # Update optimizer.pth file
    if not os.path.exists(optimizer_path):
        print(f"Error: {optimizer_path} not found.")
        return
    optimizer_data = torch.load(optimizer_path)
    print("Current learning rates in optimizer:")
    for param_group in optimizer_data['param_groups']:
        print(f"- Group with lr {param_group['lr']}")
        param_group['lr'] = lr
    torch.save(optimizer_data, optimizer_path)
    print(f"Updated optimizer learning rate to {lr}")

def main():
    if len(sys.argv) != 7:
        print("Usage: python noam_editor.py <folder_path> <batch_size> <noam_warmup> <noam_factor> <step> <lr>")
        return

    folder_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    noam_warmup = int(sys.argv[3])
    noam_factor = float(sys.argv[4])
    step = int(sys.argv[5])
    lr = float(sys.argv[6])

    print(f"Updating files in {folder_path} with batch_size={batch_size}, noam_warmup={noam_warmup}, noam_factor={noam_factor}, step={step}, lr={lr}")

    update_files(folder_path, batch_size, noam_warmup, noam_factor, step, lr)
    print("Updates applied successfully.")

if __name__ == "__main__":
    main()

