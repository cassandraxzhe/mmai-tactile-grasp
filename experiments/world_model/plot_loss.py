"""
Parse training log and plot loss curves.
Run: python plot_loss.py --log logs/11498349.out
"""
import argparse, re
import matplotlib.pyplot as plt

def main(args):
    epochs, train_losses, val_mses = [], [], []

    with open(args.log) as f:
        for line in f:
            m = re.search(r"Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_noise_mse=([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_mses.append(float(m.group(3)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, val_mses,    label="Val noise-MSE", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("World Model Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/11498349.out")
    p.add_argument("--out", default="loss_curve.png")
    main(p.parse_args())
