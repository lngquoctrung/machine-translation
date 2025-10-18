"""Helper functions"""

import pickle
import matplotlib.pyplot as plt


def save_tokenizer(tokenizer, filepath: str):
    """Save tokenizer to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath: str):
    """Load tokenizer from file"""
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {filepath}")
    return tokenizer


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
