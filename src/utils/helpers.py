"""Helper functions"""

import pickle
import os
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

