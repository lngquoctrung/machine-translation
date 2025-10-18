import numpy as np
from collections import Counter
from typing import List

class BLEUScore:
    """
    BLEU Score calculator
    Estimate BLEU score for machine translation
    """

    def __init__(self, max_n=4, weights=None):
        self.max_n = max_n
        self.weights = weights

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)

    def _modified_precision(self, reference: List[str],
                            candidate: List[str], n: int) -> float:
        """Calculate modified n-gram precision"""
        ref_ngrams = self._get_ngrams(reference, n)
        cand_ngrams = self._get_ngrams(candidate, n)

        if not cand_ngrams:
            return 0.0

        clipped_count = 0
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        total_count = sum(cand_ngrams.values())

        return clipped_count / total_count if total_count > 0 else 0.0

    def _brevity_penalty(self, reference: List[str],
                         candidate: List[str]) -> float:
        """Calculate brevity penalty"""
        ref_len = len(reference)
        cand_len = len(candidate)

        if cand_len > ref_len:
            return 1.0
        elif cand_len == 0:
            return 0.0
        else:
            return np.exp(1 - ref_len / cand_len)

    def compute(self, reference: str, candidate: str) -> float:
        """
        Compute BLEU score

        Returns:
            BLEU score (0-100)
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        # Calculate precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            p_n = self._modified_precision(ref_tokens, cand_tokens, n)
            precisions.append(p_n)

        # Geometric mean
        if min(precisions) == 0:
            geo_mean = 0.0
        else:
            log_precisions = [
                w * np.log(p) for w, p in zip(self.weights, precisions)
            ]
            geo_mean = np.exp(sum(log_precisions))

        # Brevity penalty
        bp = self._brevity_penalty(ref_tokens, cand_tokens)

        # BLEU score
        bleu = bp * geo_mean * 100

        return bleu