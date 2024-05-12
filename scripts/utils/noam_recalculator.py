import sys
import math
import torch
import os

def adjust_learning_rate_curve(original_factor, original_batch_size, current_step, current_lr, target_batch_size):
    correction_factor = original_batch_size / target_batch_size
    new_factor = original_factor * math.sqrt(correction_factor)
    new_start_step = int(current_step * correction_factor)
    return new_factor, new_start_step

def update_files(new_factor, new_start_step):
    # Update tracker.pth file
    if not os.path.exists('tracker.pth'):
        print("Error: tracker.pth not found in the current directory.")
        sys.exit(1)
    
    tracker_data = torch.load('tracker.pth')
    tracker_data['step'] = new_start_step
    torch.save(tracker_data, 'tracker_new.pth')
    
    # Update scheduler.pth file
    if not os.path.exists('scheduler.pth'):
        print("Error: scheduler.pth not found in the current directory.")
        sys.exit(1)
    
    scheduler_data = torch.load('scheduler.pth')
    scheduler_data['factor'] = round(new_factor, 1)
    scheduler_data['steps'] = new_start_step
    torch.save(scheduler_data, 'scheduler_new.pth')

def main():
    if len(sys.argv) != 6:
        print("Usage: python noam_recalculator.py <original_factor> <original_batch_size> <current_step> <target_batch_size> <current_lr>")
        sys.exit(1)
    
    original_factor = float(sys.argv[1])
    original_batch_size = int(sys.argv[2])
    current_step = int(sys.argv[3])
    target_batch_size = int(sys.argv[4])
    current_lr = float(sys.argv[5])

    new_factor, new_start_step = adjust_learning_rate_curve(original_factor, original_batch_size, current_step, current_lr, target_batch_size)
    print(f"New factor to use: {new_factor}")
    print(f"New start step to adjust learning rate curve: {new_start_step}")

    choice = input("Apply changes? (y/n): ")
    if choice.lower() == 'y':
        update_files(new_factor, new_start_step)
        print("Changes applied and files saved as tracker_new.pth and scheduler_new.pth.")
    else:
        print("No changes applied.")

if __name__ == "__main__":
    main()
