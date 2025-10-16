import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

class DataPreprocesser:
    """Handle data preprocessing for machine translation"""

    def __init__(self, max_vocab_src, max_vocab_trg):
        self.max_vocab_src = max_vocab_src
        self.max_vocab_trg = max_vocab_trg
        self.tokenizer_src = None
        self.tokenizer_trg = None

    def load_data(self, src_path, trg_path):
        """Load source and targe datasets"""
        with open(src_path, 'r', encoding='utf-8') as f:
            src_data = f.readlines()

        with open(trg_path, 'r', encoding='utf-8') as f:
            trg_data = f.readlines()

        # Add START and END tokens to the source dataset
        src_data = [f"START {text.strip()} END" for text in src_data]

        return pd.DataFrame(
            {'src': src_data,
             'trg': trg_data}
        )

    def build_tokenizer(self, data):
        """Build tokenizers for source and target datasets"""
        # Source tokenizer
        self.tokenizer_src = Tokenizer(
            num_words=self.max_vocab_src,
            filters=''
        )
        self.tokenizer_src.fit_on_texts(data['src'])

        # Target tokenizer
        self.tokenizer_trg = Tokenizer(
            num_words=self.max_vocab_trg,
            filters=''
        )
        self.tokenizer_trg.fit_on_texts(data['trg'])

        return self.tokenizer_src, self.tokenizer_trg

    def prepare_sequences(self, data, max_len_src, max_len_trg):
        """Convert text to sequences and pad"""
        # Encode
        src_sequences = self.tokenizer_src.texts_to_sequences(data['src'])
        trg_sequences = self.tokenizer_trg.texts_to_sequences(data['trg'])

        # Pad
        src_padded = pad_sequences(
            src_sequences,
            maxlen=max_len_src,
            padding='post'
        )
        trg_padded = pad_sequences(
            trg_sequences,
            maxlen=max_len_trg,
            padding='post'
        )

        # Create decoder input (shift by 1)
        trg_input = trg_padded
        trg_output = np.array([seq[1:] for seq in trg_sequences])
        trg_output = pad_sequences(
            trg_output,
            maxlen=max_len_trg,
            padding='post'
        )

        return src_padded, trg_input, trg_output

    def split_data(self, data, train_rate=0.9, val_rate=0.1):
        """Split data into train, validation and test sets"""
        train_size = int(len(data) * train_rate)
        val_size = int(len(data) * val_rate)

        train_data = data.iloc[:train_size - val_size].copy()
        val_data = data.iloc[train_size - val_size:train_size].copy()
        test_data = data.iloc[train_size:].copy()

        return train_data, val_data, test_data
