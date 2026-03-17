#!/usr/bin/env python3
#
# Production-Grade Training Log Analyzer for Fighter Aircraft Model
# Generates presentation-worthy plots for Accuracy and Learning Rate.
# Supports optional HTML report embedding the plot and optional file-based logging.
#

import argparse                     # For command-line argument parsing
import logging                      # For logging to console and file 
import re                           # For regex parsing of log files
import sys                          # For system-specific parameters and functions
from datetime import datetime       # For timestamping log files and reports
import os                           # For file path operations
from fastapi import logger
import matplotlib.pyplot as plt     # For plotting accuracy and learning rate
import seaborn as sns               # For enhanced plot aesthetics
import pandas as pd                 # For data manipulation and analysis


# Generate a timestamped filename for the plot and the program's own log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
program_name = os.path.splitext(os.path.basename(__file__))[0]
plot_filename = f"{program_name}_{timestamp}.png"


def str2bool(v):
    """Convert string representation to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_logging(generate_log: bool):
    """Set up logging to console (always) and optionally to file."""
    logger = logging.getLogger("training_plot_analyzer")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    program_log_filename = None
    if generate_log:    
        program_log_filename = f"{program_name}_{timestamp}.log"
        file_handler = logging.FileHandler(program_log_filename)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {program_log_filename}")

    return logger, program_log_filename


def parse_log_file(log_path: str, logger):
    """Parse the training log file using regex."""
    epochs, train_acc, val_acc, lr = [], [], [], []

    pattern = r"Epoch (\d+) \| Train Acc: ([\d.]+) \| Val Acc: ([\d.]+) \| LR: ([\d.e+-]+)"

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(pattern, line.strip())
                if match:
                    epochs.append(int(match.group(1)))
                    train_acc.append(float(match.group(2)))
                    val_acc.append(float(match.group(3)))
                    lr.append(float(match.group(4)))

        if not epochs:
            logger.error("No training epochs found in the log file!")
            sys.exit(1)

        logger.info(f"Successfully parsed {len(epochs)} epochs from {log_path}")

        df = pd.DataFrame({
            "Epoch": epochs,
            "Train Acc": train_acc,
            "Val Acc": val_acc,
            "LR": lr
        })

        df = df.sort_values("Epoch").reset_index(drop=True)
        return df

    except FileNotFoundError:
        logger.error(f"Log file not found: {log_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing log file: {e}")
        sys.exit(1)


def create_plots(df: pd.DataFrame, logger, generate_html: bool):
    """Generate high-quality plots for accuracy and learning rate."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "figure.figsize": (12, 10),
        "figure.dpi": 300,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(df["Epoch"], df["Train Acc"], label="Train Accuracy", linewidth=2.5, marker='o', markersize=5, alpha=0.9)
    axs[0].plot(df["Epoch"], df["Val Acc"], label="Validation Accuracy", linewidth=2.5, marker='s', markersize=5, alpha=0.9)
    axs[0].set_title("Train vs. Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, alpha=0.3, linestyle="--")

    axs[1].plot(df["Epoch"], df["LR"], label="Learning Rate", color="#d62728", linewidth=2.5, marker='^', markersize=5)
    axs[1].set_title("Learning Rate Schedule (Cosine Annealing)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Learning Rate")
    axs[1].set_yscale("log")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved as: {plot_filename}")

    if not generate_html:
        plt.show()
    else:
        plt.close(fig)


def generate_html_report(df: pd.DataFrame, logger, input_log_path: str, program_log_filename: str | None = None):
    """Generate HTML report showing the processed input log file."""
    html_filename = "training_report.html"

    best_val_acc = df["Val Acc"].max()
    best_epoch = df.loc[df["Val Acc"].idxmax(), "Epoch"]
    num_epochs = len(df)

    log_info_lines = []
    log_info_lines.append(f"<li><strong>Training Log File:</strong> {input_log_path}</li>")

    if program_log_filename:
        log_info_lines.append(f"<li><strong>Report Log File:</strong> {program_log_filename}</li>")
    else:
        log_info_lines.append("<li><em>No separate analysis log file generated (console only)</em></li>")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fighter Aircraft Model Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2 {{ color: #2c3e50; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        .metrics {{ background: #f9f9f9; padding: 20px; border-radius: 8px; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>Fighter Aircraft Model Training Report</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Training Progress</h2>
    <img src="{plot_filename}" alt="Train/Val Accuracy and Learning Rate Plot">

    <div class="metrics">
        <h3>Key Metrics</h3>
        <ul>
            {''.join(log_info_lines)}
            <li><strong>Best Validation Accuracy:</strong> {best_val_acc:.4f} at Epoch {best_epoch}</li>
            <li><strong>Total Epochs:</strong> {num_epochs}</li>
            <li><strong>Final Training Accuracy:</strong> {df["Train Acc"].iloc[-1]:.4f}</li>
            <li><strong>Final Validation Accuracy:</strong> {df["Val Acc"].iloc[-1]:.4f}</li>
            <li><strong>Final Learning Rate:</strong> {df["LR"].iloc[-1]:.2e}</li>
        </ul>
    </div>

    <p><em>Tip from your ML Evangelist:</em> Watch for overfitting if train acc >> val acc. Consider early stopping or stronger regularization if the gap widens significantly.</p>

    <footer>
        <p>Generated with ❤️ by the enhanced training plotter program</p>
    </footer>
</body>
</html>
"""

    try:
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report generated: {html_filename}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Production-grade analyzer for fighter aircraft model training logs. "
                    "Generates plots and optional HTML report."
    )
    parser.add_argument("--log", type=str, default="fighter_id11.log",
                        help="Path to the training log file")
    parser.add_argument("--generate-html", type=str2bool, default=True,
                        help="Generate HTML report with embedded plot (default: True)")
    parser.add_argument("--generate-log", type=str2bool, default=True,
                        help="Generate a timestamped log file in addition to console output (default: True)")

    args = parser.parse_args()

    logger, program_log_filename = setup_logging(args.generate_log)

    logger.info(f"Analyzing log file: {args.log}")
    logger.info(f"HTML report: {'enabled' if args.generate_html else 'disabled'}")
    logger.info(f"File logging: {'enabled' if args.generate_log else 'disabled'}")

    try:
        df = parse_log_file(args.log, logger)
        create_plots(df, logger, generate_html=args.generate_html)

        if args.generate_html:
            generate_html_report(
                df,
                logger,
                input_log_path=args.log,
                program_log_filename=program_log_filename
            )

        # Final summary
        best_val_idx = df["Val Acc"].idxmax()
        logger.info("Final metrics summary:")
        logger.info(f"   Best Val Acc : {df['Val Acc'].iloc[best_val_idx]:.4f} at epoch {df['Epoch'].iloc[best_val_idx]}")
        logger.info(f"   Final Train Acc : {df['Train Acc'].iloc[-1]:.4f}")
        logger.info(f"   Final Val Acc : {df['Val Acc'].iloc[-1]:.4f}")
        logger.info(f"   Final LR : {df['LR'].iloc[-1]:.2e}")
        logger.info("Analysis complete. Happy training! 🚀")

    except Exception as e:
        logger.error(f"Failed during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()