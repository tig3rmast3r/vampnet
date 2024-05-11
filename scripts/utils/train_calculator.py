# noamcalculator.py
import sys

def noam_learning_rate(step, d_model, warmup_steps, factor, base_lr):
    """Calculates total training"""
    scale = factor * d_model ** -0.5
    warmup_factor = step * (warmup_steps ** -1.5)
    decay_factor = step ** -0.5
    noam_lr = scale * min(warmup_factor, decay_factor)
    return noam_lr * base_lr

def main():
    if len(sys.argv) != 7:
        print("Usage: python noamcalculator.py <Noam.warmup> <total_steps> <batch_size> <embedding_dim> <Noam.factor> <adamw_lr>")
        return

    try:
        warmup_steps = int(sys.argv[1])
        total_steps = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        d_model = int(sys.argv[4])
        factor = float(sys.argv[5])
        adamw_lr = float(sys.argv[6])
    except ValueError:
        print("Error: warmup_steps, total_steps, e batch_size must be Int. Factor and adamw_lr must be numbers.")
        return

    # Calcolare il learning rate massimo al termine del warmup
    max_lr = noam_learning_rate(min(warmup_steps, total_steps), d_model, warmup_steps, factor, adamw_lr)

    # Calcolare la somma totale dei learning rate durante tutte le iterazioni
    total_lr_sum = sum(noam_learning_rate(step, d_model, warmup_steps, factor, adamw_lr) for step in range(1, total_steps + 1))

    # Considera l'effetto del batch size sul totale
    total_lr_sum_adjusted = total_lr_sum * batch_size

    # Stampa i risultati
    print(f"Max learning rate at warmup ({warmup_steps} steps): {max_lr}")
    print(f"Total Learning rate (adjusted for batch size {batch_size}) during training ({total_steps} steps): {total_lr_sum_adjusted}")

if __name__ == "__main__":
    main()

