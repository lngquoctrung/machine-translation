"""Helper functions"""

import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_ROOT_DIR = Path(__file__).resolve().parents[2]

def save_tokenizer(tokenizer, filepath: str):
    """Save tokenizer to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {sanitize_path(filepath)}")


def load_tokenizer(filepath: str):
    """Load tokenizer from file"""
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {sanitize_path(filepath)}")
    return tokenizer

def sanitize_path(path: str, root_dir: Path = DEFAULT_ROOT_DIR) -> str:
    """
    Remove or anonymize the base part of a path to avoid exposing absolute directories.

    Parameters:
        path (str): The path of a file or directory.
        root_dir (Path): Root directory to anonymize. Defaults to project root.

    Returns:
        str: Sanitized path
    """
    try:
        rel_path = os.path.relpath(path, start=str(root_dir))
        return os.path.join("...", rel_path)
    except Exception:
        return os.path.join("...", os.path.basename(path))


def plot_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
