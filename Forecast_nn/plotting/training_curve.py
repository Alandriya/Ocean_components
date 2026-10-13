"""Plot training history saved by Forecast_nn.train."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_history(path):
    with np.load(path) as f:
        return {key: np.asarray(f[key]) for key in f.files}


def plot_training_curve(input_path, output_path=None, title=None):
    history = load_history(input_path)
    epochs = history.get('epoch', np.arange(1, len(history['total_loss']) + 1))
    total = history['total_loss']

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(epochs, total, linewidth=2, label='Total loss')
    if 'mse' in history and len(history['mse']) == len(epochs):
        ax.plot(epochs, history['mse'], linewidth=1.8, label='MSE')
    if 'dynamic' in history and len(history['dynamic']) == len(epochs) and np.any(np.asarray(history['dynamic']) != 0):
        ax.plot(epochs, history['dynamic'], linewidth=1.4, linestyle='--', label='Dynamic loss')
    if 'sde' in history and len(history['sde']) == len(epochs) and np.any(np.asarray(history['sde']) != 0):
        ax.plot(epochs, history['sde'], linewidth=1.4, linestyle='--', label='SDE energy')
    if 'sde_drift' in history and len(history['sde_drift']) == len(epochs) and np.any(np.asarray(history['sde_drift']) != 0):
        ax.plot(epochs, history['sde_drift'], linewidth=1.2, linestyle=':', label='SDE drift MSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / diagnostic')
    ax.set_title(title or 'Training curve')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path or Path(input_path).with_suffix('.png'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default=None)
    p.add_argument('--title', default=None)
    args = p.parse_args()
    saved = plot_training_curve(args.input, args.output, args.title)
    print(f'Saved: {saved}')


if __name__ == '__main__':
    main()
