import numpy as np
import re
import contractions
import string  # ← THÊM IMPORT
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BeamSearchDecoder:
    """
    Beam Search Decoder with contraction expansion and punctuation removal
    - Detect proper names (capitalized words)
    - Replace with placeholders before translation
    - Restore after decoding
    """
    
    def __init__(self, model, tokenizer_src, tokenizer_trg, max_len_src, max_len_trg,
                 beam_width=5, expand_contractions=True, remove_punctuation=True):
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.max_len_src = max_len_src
        self.max_len_trg = max_len_trg
        self.beam_width = beam_width
        self.expand_contractions = expand_contractions
        self.remove_punctuation = remove_punctuation
        self.start_token = tokenizer_trg.word_index.get("start", 1)
        self.end_token = tokenizer_trg.word_index.get("end", 2)

    def _extract_proper_names(self, text: str) -> dict:
        """
        Extract proper name and check if it is in vocabulary
        """
        # Expand contractions trước
        if self.expand_contractions:
            text = contractions.fix(text)

        # Remove punctuation
        clean_text = text.translate(str.maketrans("", "", string.punctuation))
        words = clean_text.split()

        proper_names = {}

        for i, word in enumerate(words):
            # Check if capitalized
            if word and word[0].isupper():
                # Check if in vocabulary
                word_lower = word.lower()
                token_id = self.tokenizer_src.word_index.get(word_lower, 0)

                if token_id == 0:
                    proper_names[i] = word

        return proper_names, words

    def preprocess(self, text: str) -> np.ndarray:
        """Preprocess input text"""
        if self.expand_contractions:
            text = contractions.fix(text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        text = text.lower()

        sequences = self.tokenizer_src.texts_to_sequences([text])
        padded = pad_sequences(
            sequences,
            maxlen=self.max_len_src,
            padding="post",
            truncating="post"
        )

        return padded

    def decode_greedy(self, text: str) -> str:
        """Greedy decoding với copy-through"""
        # Extract OOV proper names
        proper_names, input_words = self._extract_proper_names(text)

        # Preprocess
        src_seq = self.preprocess(text)

        # Initialize
        target_seq = np.zeros((1, self.max_len_trg))
        target_seq[0, 0] = self.start_token

        translated_tokens = []

        for i in range(1, self.max_len_trg):
            predictions = self.model.predict([src_seq, target_seq], verbose=0)
            predicted_id = np.argmax(predictions[0, i-1, :])

            if predicted_id == self.end_token or predicted_id == 0:
                break

            translated_tokens.append(predicted_id)
            target_seq[0, i] = predicted_id

        # Convert to text
        translated_text = self._tokens_to_text(translated_tokens)

        # Copy-through OOV proper names
        translated_text = self._copy_through_names(
            translated_text, 
            proper_names, 
            input_words
        )

        return translated_text

    def decode_beam_search(self, text: str) -> str:
        """Beam search with copy-through"""
        proper_names, input_words = self._extract_proper_names(text)

        src_seq = self.preprocess(text)

        beams = [(0.0, [self.start_token])]

        for step in range(self.max_len_trg - 1):
            new_beams = []

            for score, seq in beams:
                if seq[-1] == self.end_token:
                    new_beams.append((score, seq))
                    continue

                target_seq = np.zeros((1, self.max_len_trg))
                for i, token_id in enumerate(seq):
                    if i < self.max_len_trg:
                        target_seq[0, i] = token_id

                predictions = self.model.predict([src_seq, target_seq], verbose=0)
                next_token_probs = predictions[0, len(seq)-1, :]

                top_indices = np.argsort(next_token_probs)[-self.beam_width:]

                for idx in top_indices:
                    if idx == 0:
                        continue
                    new_score = score - np.log(next_token_probs[idx] + 1e-10)
                    new_seq = seq + [idx]
                    new_beams.append((new_score, new_seq))

            beams = sorted(new_beams, key=lambda x: x[0])[:self.beam_width]

            if all(seq[-1] == self.end_token for _, seq in beams):
                break

        best_seq = beams[0][1][1:]
        if best_seq and best_seq[-1] == self.end_token:
            best_seq = best_seq[:-1]

        translated_text = self._tokens_to_text(best_seq)
        translated_text = self._copy_through_names(
            translated_text,
            proper_names,
            input_words
        )

        return translated_text

    def _tokens_to_text(self, tokens: list) -> str:
        """Convert tokens to text"""
        words = []
        index_word = {v: k for k, v in self.tokenizer_trg.word_index.items()}

        for token_id in tokens:
            if token_id in index_word:
                word = index_word[token_id]
                if word not in ["start", "end"]:
                    words.append(word)

        return " ".join(words)

    def _copy_through_names(self, translated_text: str, proper_names: dict, 
                            input_words: list) -> str:
        """
        Copy-through mechanism
        """
        if not proper_names:
            return translated_text

        # Split và tìm <UNK>
        output_words = translated_text.split()
        sorted_names = [proper_names[pos] for pos in sorted(proper_names.keys())]
        name_idx = 0
        for i, word in enumerate(output_words):
            if word == "<UNK>" and name_idx < len(sorted_names):
                output_words[i] = sorted_names[name_idx]
                name_idx += 1

        return " ".join(output_words)