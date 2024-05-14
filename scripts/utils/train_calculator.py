import sys

def noam_learning_rate(step, d_model, warmup_steps, factor):
    """Calcola il learning rate usando la formula del Noam Scheduler senza base_lr."""
    scale = factor * (d_model ** -0.5)
    warmup_factor = step * (warmup_steps ** -1.5)
    decay_factor = step ** -0.5
    return scale * min(warmup_factor, decay_factor)

def main():
    if len(sys.argv) != 6:
        print("Usage: python noamcalculator.py <warmup_steps> <total_steps> <batch_size> <d_model> <factor>")
        return

    try:
        warmup_steps = int(sys.argv[1])
        total_steps = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        d_model = int(sys.argv[4])
        factor = float(sys.argv[5])
    except ValueError:
        print("Error: All inputs must be numbers, with batch_size, warmup_steps, total_steps, and d_model as integers.")
        return

    # Calculate the learning rate at the end of warmup and at a specific step
    max_lr = noam_learning_rate(min(warmup_steps, total_steps), d_model, warmup_steps, factor)

    # Calculate the total sum of learning rates during all iterations
    total_lr_sum = sum(noam_learning_rate(step, d_model, warmup_steps, factor) for step in range(1, total_steps + 1))

    # Adjust the total sum for batch size
    total_lr_sum_adjusted = total_lr_sum * batch_size

    # Print the results
    print(f"Max learning rate at warmup ({warmup_steps} steps): {max_lr}")
    print(f"Total cumulative learning rate for batch size {batch_size} during training ({total_steps} steps): {total_lr_sum_adjusted}")

if __name__ == "__main__":
    main()
