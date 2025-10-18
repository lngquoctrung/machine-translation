import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from collections import Counter

from src.utils import setup_logger

class DataPreprocessor:
    """
    Handle data preprocessing for machine translation
        - Filter rare words to reduce vocabulary size
        - Filter long sentences to save memory
        - Tokenization and padding
    """

    def __init__(self, max_vocab_src, max_vocab_trg, min_frequency=2, name_logger=__name__, filename_logger=None):
        self.max_vocab_src = max_vocab_src
        self.max_vocab_trg = max_vocab_trg
        self.min_frequency = min_frequency
        self.tokenizer_src = None
        self.tokenizer_trg = None
        self.logger = setup_logger(
            name=name_logger,
            log_file=filename_logger
        )

    def load_data(self, src_path, trg_path, max_length_src=40, max_length_trg=50):
        """
        Load source and targe datasets and filter by lenght

        Parameters:
            src_path: path to source dataset
            trg_path: path to target dataset
            max_length_src: maximum source sequence length
            max_length_trg: maximum target sequence length
        """
        with open(src_path, 'r', encoding='utf-8') as f:
            src_data = f.readlines()
        with open(trg_path, 'r', encoding='utf-8') as f:
            trg_data = f.readlines()

        # Filter too long sentences
        filtered_data = []
        for src_line, trg_line in zip(src_data, trg_data):
            src_words = src_line.strip().split()
            trg_words = trg_line.strip().split()
            if len(src_words) < max_length_src and len(trg_words) < max_length_trg:
                filtered_data.append((src_line.strip(), trg_line.strip()))

        self.logger.info(f"Filtered: {len(filtered_data)/len(src_data)} pairs kept")
        self.logger.info(f"Memory save: {(1 - len(filtered_data)/len(src_data)) * 100:.1f}%")

        # Add START and END tokens to the source dataset
        trg_processed = [f"START {pair[1].strip()} END" for pair in filtered_data]

        return pd.DataFrame({
            'src': [pair[0] for pair in filtered_data],
            'trg': trg_processed
        })

    def filter_rare_words(self, texts, min_frequency=2):
        """
        Filter rare words to reduce vocabulary size

        Parameters:
            texts: list of strings
            min_frequency: minimum frequency of rare words
        """
        word_freq = Counter()
        for text in texts:
            word_freq.update(text.lower().split())

        filtered_texts = []
        for text in texts:
            words = text.lower().split()
            filtered_words = [
                w if word_freq[w] >= min_frequency else "<UNK>"
                for w in words
            ]
            filtered_texts.append(" ".join(filtered_words))

        vocab_size_before = len(word_freq)
        vocab_size_after = len([w for w, c in word_freq.items() if c >= min_frequency])

        self.logger.info(f"Vocab reduced: {vocab_size_before} → {vocab_size_after}")
        self.logger.info(f"Reduction: {(1 - vocab_size_after / vocab_size_before) * 100:.1f}%")

        return filtered_texts

    def build_tokenizers(self, data):
        """Build tokenizers for source and target datasets"""
        self.logger.info("Filtering rare words in source dataset...")
        src_filtered = self.filter_rare_words(
            data['src'],
            self.min_frequency
        )
        self.logger.info("Filtering rare words in target dataset...")
        trg_filtered = self.filter_rare_words(
            data['trg'],
            self.min_frequency
        )

        # Source tokenizer
        self.tokenizer_src = Tokenizer(
            num_words=self.max_vocab_src,
            oov_token="<UNK>",
            filters=""
        )
        self.tokenizer_src.fit_on_texts(src_filtered)

        # Target tokenizer
        self.tokenizer_trg = Tokenizer(
            num_words=self.max_vocab_trg,
            oov_token="<UNK>",
            filters=""
        )
        self.tokenizer_trg.fit_on_texts(trg_filtered)

        return self.tokenizer_src, self.tokenizer_trg

    def prepare_sequences(self, data, max_len_src, max_len_trg):
        """Convert text to sequences with padding"""
        # Encode
        src_sequences = self.tokenizer_src.texts_to_sequences(data['src'])
        trg_sequences = self.tokenizer_trg.texts_to_sequences(data['trg'])

        # Pad
        src_padded = pad_sequences(
            src_sequences,
            maxlen=max_len_src,
            padding='post',
            truncating='post',
        )
        trg_padded = pad_sequences(
            trg_sequences,
            maxlen=max_len_trg,
            padding='post',
            truncating='post',
        )

        # Create decoder input (shift by 1)
        trg_input = trg_padded
        trg_output = np.array([seq[1:] for seq in trg_sequences])
        trg_output = pad_sequences(
            trg_output,
            maxlen=max_len_trg,
            padding='post',
            truncating='post',
        )

        # Memory info
        memory_mb = (
                src_padded.nbytes + trg_input.nbytes + trg_output.nbytes
        ) / (1024 * 1024)
        self.logger.info(f"Sequences memory: {memory_mb:.2f} MB")

        return src_padded, trg_input, trg_output

    def split_data(self, data, train_rate=0.9, val_rate=0.1):
        """Split data into train, validation and tests sets"""
        train_size = int(len(data) * train_rate)
        val_size = int(len(data) * val_rate)

        train_data = data.iloc[:train_size - val_size].copy()
        val_data = data.iloc[train_size - val_size:train_size].copy()
        test_data = data.iloc[train_size:].copy()

        return train_data, val_data, test_data
