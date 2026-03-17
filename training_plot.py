import argparse
import logging
import re
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# =============================================================================
# Production-Grade Log Analyzer for Fighter Aircraft Training
# Generates presentation-worthy plots: Accuracy + Learning Rate
# =============================================================================

def setup_logging():
    logger = logging.getLogger("log_analyzer")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def parse_log_file(log_path: str, logger):
    epochs = []
    train_acc = []
    val_acc = []
    lr = []

    pattern = r"Epoch (\d+) \| Train Acc: ([\d.]+) \| Val Acc: ([\d.]+) \| LR: ([\d.]+)"

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    epoch = int(match.group(1))
                    t_acc = float(match.group(2))
                    v_acc = float(match.group(3))
                    l_rate = float(match.group(4))

                    epochs.append(epoch)
                    train_acc.append(t_acc)
                    val_acc.append(v_acc)
                    lr.append(l_rate)

        if not epochs:
            logger.error("No training epochs found in log file!")
            sys.exit(1)

        logger.info(f"✅ Parsed {len(epochs)} epochs successfully")
        return pd.DataFrame({
            "Epoch": epochs,
            "Train Acc": train_acc,
            "Val Acc": val_acc,
            "LR": lr
        })

    except FileNotFoundError:
        logger.error(f"Log file not found: {log_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing log: {e}")
        sys.exit(1)


def create_plots(df: pd.DataFrame, logger):
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

    # Accuracy Plot
    axs[0].plot(df["Epoch"], df["Train Acc"], label="Train Accuracy", linewidth=2.5, marker='o', markersize=4)
    axs[0].plot(df["Epoch"], df["Val Acc"],   label="Val Accuracy",   linewidth=2.5, marker='s', markersize=4)
    axs[0].set_title("Train vs Validation Accuracy", fontsize=16, fontweight="bold")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Learning Rate Plot
    axs[1].plot(df["Epoch"], df["LR"], label="Learning Rate", color="#d62728", linewidth=2.5, marker='^', markersize=4)
    axs[1].set_title("Learning Rate Schedule (Cosine Annealing)", fontsize=16, fontweight="bold")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Learning Rate")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    axs[1].set_yscale("log")  # LR is best viewed on log scale

    plt.tight_layout()
    plt.savefig("training_plot.png", dpi=300, bbox_inches="tight")
    # plt.savefig("lr_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info("✅ Plot saved as:")
    logger.info("   • training_plot.png (Train/Val Accuracy) and (Learning Rate)")
    # logger.info("   • lr_plot.png (Learning Rate)")


def main():
    parser = argparse.ArgumentParser(description="Production-grade Training Log Analyzer")
    parser.add_argument("--log", type=str, default="fighter_id10.log",
                        help="Path to the training log file")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Analyzing log: {args.log}")

    df = parse_log_file(args.log, logger)
    create_plots(df, logger)

    logger.info(f"Final metrics from log:")
    logger.info(f"   Best Val Acc : {df['Val Acc'].max():.4f} at epoch {df.loc[df['Val Acc'].idxmax(), 'Epoch']}")
    logger.info(f"   Final Test Acc would be shown in the original log")


if __name__ == "__main__":
    main()
    