"""Byte-level BPE training and tokenization for CS336 Assignment 1.

The public API deliberately follows the data structures used by the assignment:
vocabularies map integer IDs to byte strings, while merges are ordered pairs of
byte strings. Internally, token IDs are used so merging stays inexpensive.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

GPT2_PRETOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _merge_token_ids(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of ``pair`` from left to right."""
    merged: list[int] = []
    index = 0
    while index < len(token_ids):
        if index + 1 < len(token_ids) and (token_ids[index], token_ids[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


def _ordinary_fragments(text: str, special_tokens: list[str]) -> Iterator[str]:
    """Yield text regions outside special tokens.

    Sorting longest-first makes overlapping special tokens deterministic.
    Special tokens are separators during training and are therefore omitted.
    """
    if not special_tokens:
        yield text
        return

    alternatives = "|".join(re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True))
    yield from re.split(alternatives, text)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on ``input_path``.

    Pair counts are updated only for pre-tokens affected by the selected merge.
    This avoids rescanning the complete corpus after every merge while preserving
    the assignment's lexicographic tie-breaking rule.
    """
    special_tokens = list(dict.fromkeys(special_tokens or []))
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

    text = Path(input_path).read_text(encoding="utf-8")

    pretoken_counts: Counter[bytes] = Counter()
    for fragment in _ordinary_fragments(text, special_tokens):
        pretoken_counts.update(match.group(0).encode("utf-8") for match in GPT2_PRETOKEN_PATTERN.finditer(fragment))

    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # Each distinct pre-token is stored once and weighted by its corpus count.
    words = [list(pretoken) for pretoken in pretoken_counts]
    frequencies = [pretoken_counts[pretoken] for pretoken in pretoken_counts]

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for word_index, (word, frequency) in enumerate(zip(words, frequencies)):
        local_counts = Counter(itertools.pairwise(word))
        for pair, occurrences in local_counts.items():
            pair_counts[pair] += frequency * occurrences
            pair_to_words[pair].add(word_index)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        new_id = len(vocab)
        left_bytes, right_bytes = vocab[best_pair[0]], vocab[best_pair[1]]
        vocab[new_id] = left_bytes + right_bytes
        merges.append((left_bytes, right_bytes))

        affected_words = tuple(pair_to_words.get(best_pair, ()))
        for word_index in affected_words:
            old_word = words[word_index]
            frequency = frequencies[word_index]
            old_counts = Counter(itertools.pairwise(old_word))

            for pair, occurrences in old_counts.items():
                pair_counts[pair] -= frequency * occurrences
                pair_to_words[pair].discard(word_index)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    pair_to_words.pop(pair, None)

            new_word = _merge_token_ids(old_word, best_pair, new_id)
            words[word_index] = new_word
            new_counts = Counter(itertools.pairwise(new_word))
            for pair, occurrences in new_counts.items():
                pair_counts[pair] += frequency * occurrences
                pair_to_words[pair].add(word_index)

    return vocab, merges


class BPETokenizer:
    """A byte-level BPE tokenizer constructed from an assignment vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.special_tokens = sorted(set(special_tokens or []), key=len, reverse=True)

        self.merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left_bytes, right_bytes) in enumerate(merges):
            left_id = self.bytes_to_id[left_bytes]
            right_id = self.bytes_to_id[right_bytes]
            merged_id = self.bytes_to_id[left_bytes + right_bytes]
            self.merge_ranks[(left_id, right_id)] = (rank, merged_id)

        self.special_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                raise ValueError(f"special token is missing from vocabulary: {token!r}")
            self.special_to_id[token] = self.bytes_to_id[token_bytes]

        if self.special_tokens:
            alternatives = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_pattern = re.compile(f"({alternatives})")
        else:
            self.special_pattern = None

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        # Assignment vocabularies do not require the token ID of a byte to equal
        # its integer value (the GPT-2 fixture uses a permuted ID assignment).
        token_ids = [self.bytes_to_id[bytes([byte])] for byte in pretoken]
        while len(token_ids) >= 2:
            candidates = {
                pair: self.merge_ranks[pair] for pair in itertools.pairwise(token_ids) if pair in self.merge_ranks
            }
            if not candidates:
                break
            best_pair = min(candidates, key=lambda pair: candidates[pair][0])
            _, merged_id = candidates[best_pair]
            token_ids = _merge_token_ids(token_ids, best_pair, merged_id)
        return token_ids

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in GPT2_PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group(0).encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))

        encoded: list[int] = []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.special_to_id:
                encoded.append(self.special_to_id[part])
            else:
                encoded.extend(self._encode_ordinary(part))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        try:
            text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return text_bytes.decode("utf-8", errors="replace")
