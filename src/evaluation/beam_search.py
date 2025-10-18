import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class BeamSearchDecoder:
    """
    Beam Search Decoder
    """

    def __init__(self, model, tokenizer_src, tokenizer_trg, max_len_src, max_len_trg, beam_width=5):
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.max_len_src = max_len_src
        self.max_len_trg = max_len_trg
        self.beam_width = beam_width

        self.start_token = tokenizer_trg.word_index.get("start", 1)
        self.end_token = tokenizer_trg.word_index.get("end", 2)

    def preprocess(self, text: str) -> np.ndarray:
        """Preprocess input text"""
        text = text.lower().strip()
        seq = self.tokenizer_src.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len_src, padding='post')
        return padded

    def decode_greedy(self, input_seq: np.ndarray) -> str:
        """Greedy decoding"""
        decoder_input = np.zeros((1, self.max_len_trg))
        decoder_input[0, 0] = self.start_token

        output_sentence = []

        for i in range(1, self.max_len_trg):
            predictions = self.model.predict(
                [input_seq, decoder_input],
                verbose=0
            )

            predicted_id = np.argmax(predictions[0, i - 1, :])

            if predicted_id == self.end_token or predicted_id == 0:
                break

            predicted_word = self.tokenizer_trg.index_word.get(predicted_id, '')
            if predicted_word and predicted_word not in ['start', 'end']:
                output_sentence.append(predicted_word)

            decoder_input[0, i] = predicted_id

        return ' '.join(output_sentence)

    def decode_beam_search(self, input_seq: np.ndarray) -> tuple:
        """Beam search decoding"""
        # Initialize beams: (sequence, score)
        beams = [(np.array([self.start_token]), 0.0)]
        completed_beams = []

        for step in range(self.max_len_trg - 1):
            all_candidates = []

            for sequence, score in beams:
                # Tạo decoder input từ sequence
                decoder_input = np.zeros((1, self.max_len_trg))
                for idx, token in enumerate(sequence):
                    decoder_input[0, idx] = token

                # Predict
                predictions = self.model.predict([input_seq, decoder_input], verbose=0)
                token_probs = predictions[0, len(sequence) - 1, :]  # Probabilities for next token

                # Get top-k candidates
                top_k_indices = np.argsort(token_probs)[-self.beam_width:]

                for token_id in top_k_indices:
                    # Calculate new score (log probability)
                    token_prob = token_probs[token_id]
                    new_score = score + np.log(token_prob + 1e-10)

                    # Create new sequence
                    new_sequence = np.append(sequence, token_id)

                    # Check if completed
                    if token_id == self.end_token or len(new_sequence) >= self.max_len_trg:
                        completed_beams.append((new_sequence, new_score))
                    else:
                        all_candidates.append((new_sequence, new_score))

            # Sort by score and keep top-k beams
            if not all_candidates:
                break

            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]

        # Add remaining beams to completed
        completed_beams.extend(beams)

        # Sort by normalized score (divide by length to avoid favoring short sequences)
        completed_beams = sorted(
            completed_beams,
            key=lambda x: x[1] / len(x[0]),
            reverse=True
        )

        # Convert sequences to text
        results = []
        for sequence, score in completed_beams[:self.beam_width]:
            words = []
            for token_id in sequence:
                if token_id == self.start_token or token_id == self.end_token:
                    continue
                word = self.tokenizer_trg.index_word.get(token_id, '')
                if word:
                    words.append(word)

            translation = ' '.join(words)
            if translation:
                results.append((translation, score / len(sequence)))

        best_translation = results[0][0] if results else ""
        return best_translation, results

    def translate(self, text: str, use_beam_search: bool = False) -> str:
        """
        Main translation method

        Args:
            text: Input English text
            use_beam_search: Use beam search (True) or greedy (False)
        """
        input_seq = self.preprocess(text)

        if use_beam_search:
            translation, _ = self.decode_beam_search(input_seq)
            return translation
        else:
            return self.decode_greedy(input_seq)