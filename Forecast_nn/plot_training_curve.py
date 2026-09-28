# Plot an SDE-Energy training curve from an existing .npy or the richer .npz history.
# Examples from the repository root:
# python -m Forecast_nn.plot_training_curve --variable sensible --downsample 2
# python -m Forecast_nn.plot_training_curve --input "D:/Nastya/Data/OceanFull/Losses/loss_SDE-Energy_sensible_ds2.npy"
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from Forecast_nn.config import cfg


def load_history(path):
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        return {key: np.asarray(data[key], dtype=float) for key in data.files}
    return {'total_loss': np.asarray(data, dtype=float)}


def plot_training_curve(input_path, output_path=None, title=None):
    history = load_history(input_path)
    total = history['total_loss']
    epochs = np.arange(1, len(total) + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, total, linewidth=2, label='Total loss')
    if 'mse' in history and len(history['mse']) == len(total):
        ax.plot(epochs, history['mse'], linewidth=1.7, label='MSE')
    if 'sde' in history and len(history['sde']) == len(total):
        # SDE can be on a very different numerical scale; show it on a secondary axis.
        ax2 = ax.twinx()
        ax2.plot(epochs, history['sde'], linewidth=1.5, linestyle='--', alpha=0.8, label='SDE term')
        ax2.set_ylabel('SDE term')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax.legend(loc='best')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title or 'SDE-Energy training curve')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(1, max(len(total), 1))
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.png'
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, help='Existing .npy total-loss file or .npz detailed training history.')
    parser.add_argument('--output', default=None)
    parser.add_argument('--variable', choices=('sensible', 'latent'), default='sensible')
    parser.add_argument('--downsample', type=int, default=2)
    args = parser.parse_args()

    if args.input is None:
        rich = os.path.join(cfg.root_path, 'Losses', f'training_history_SDE-Energy_{args.variable}_ds{args.downsample}.npz')
        legacy = os.path.join(cfg.root_path, 'Losses', f'loss_SDE-Energy_{args.variable}_ds{args.downsample}.npy')
        input_path = rich if os.path.exists(rich) else legacy
    else:
        input_path = args.input

    output_path = args.output or os.path.join(cfg.root_path, 'videos/Forecast/Losses', f'training_curve_SDE-Energy_{args.variable}_ds{args.downsample}.png')
    saved = plot_training_curve(input_path, output_path, title=f'SDE-Energy training: {args.variable}, ds={args.downsample}')
    print(f'Training curve saved to: {saved}')


if __name__ == '__main__':
    main()
