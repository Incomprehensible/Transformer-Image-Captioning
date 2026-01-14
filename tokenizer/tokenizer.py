import json
import os
import re
from typing import List, Tuple
from collections import Counter
import sys
import config

sys.path.append("..")

from .utils import (
    byte_to_unicode,
    unicode_to_byte_map,
    get_pairs,
    get_stats,
    merge_vocab
)

class ByteLevelBPE:

    def __init__(self, special_tokens: List[config.SpecialTokens] = None):
        self.byte_encoder = byte_to_unicode()
        self.byte_decoder = unicode_to_byte_map()
        self.bpe_ranks = {}
        self.encoder = {}
        self.decoder = {}
        self.special_tokens = special_tokens if special_tokens else []
        self.cache = {}

        self.pat = re.compile(
            r"'(?:s|t|re|ve|m|ll|d)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+",
            re.IGNORECASE
        )

    def train(self, texts: List[str], vocab_size: int = 1000, verbose: bool = False):

        special_tokens_len = len(self.special_tokens)
        if vocab_size - special_tokens_len < 256:
            raise ValueError(
                f"vocab_size (without special tokens) must be at least 256, got {vocab_size}"
            )

        special_tokens_dict = {
            self.special_tokens[i]: i for i in range(len(self.special_tokens))
        }

        word_freqs = Counter()

        for text in texts:
            tokens = re.findall(self.pat, text)
            for token in tokens:
                token_bytes = token.encode("utf-8")
                byte_encoded = "".join(self.byte_encoder[b] for b in token_bytes)
                word_freqs[byte_encoded] += 1

        vocab = {tuple(word): freq for word, freq in word_freqs.items()}

        num_merges = vocab_size - 256 - special_tokens_len
        merges = []

        for i in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                if verbose:
                    print(f"\nWarning: No more pairs to merge at iteration {i}")
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = merge_vocab(best_pair, vocab)
            merges.append(best_pair)

        self.encoder = {
            self.byte_encoder[i]: i + special_tokens_len for i in range(256)
        }
        self.encoder.update(special_tokens_dict)

        next_id = 256 + special_tokens_len
        for pair in merges:
            merged_token = "".join(pair)
            self.encoder[merged_token] = next_id
            next_id += 1

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        if verbose:
            print(f"\nVocab size (with special tokens): {len(self.encoder)}")
            print(f"Made {len(merges)} merges")

    def _apply_merge(self, word: tuple, pair: Tuple[str, str]) -> tuple:
        first, second = pair
        new_word = []
        i = 0

        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda p: self.bpe_ranks.get(p, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            word = self._apply_merge(word, bigram)
            if len(word) == 1:
                break

            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def tokenize(self, text: str) -> List[str]:
        bpe_tokens = []

        for raw_token in re.findall(self.pat, text):
            token_bytes = raw_token.encode("utf-8")
            byte_token = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_result = self.bpe(byte_token)
            bpe_tokens.extend(bpe_result.split(" "))

        return bpe_tokens

    def encode(
        self,
        text: str,
        max_seq_length: int | None = None,
        verbose: bool = False
    ) -> List[int]:

        tokens = self.tokenize(text)
        ids = [self.encoder.get(token, 0) for token in tokens]

        # Add BOS / EOS
        if config.SpecialTokens.BOS in self.special_tokens:
            ids = [self.encoder[config.SpecialTokens.BOS]] + ids
        if config.SpecialTokens.EOS in self.special_tokens:
            ids = ids + [self.encoder[config.SpecialTokens.EOS]]

        if max_seq_length is not None:
            if len(ids) > max_seq_length:
                ids = ids[:max_seq_length]
                if verbose:
                    print(
                        f"Warning: input text truncated to {max_seq_length} tokens."
                    )

            if config.SpecialTokens.PAD in self.special_tokens:
                pad_id = self.encoder[config.SpecialTokens.PAD]
                ids = ids + [pad_id] * (max_seq_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []

        for token_id in ids:
            token_str = self.decoder.get(token_id, "")
            if token_str in self.special_tokens:
                continue
            tokens.append(token_str)

        text = "".join(tokens)
        byte_list = bytearray()

        for char in text:
            if char in self.byte_decoder:
                byte_list.append(self.byte_decoder[char])
            else:
                byte_list.append(ord("?"))

        return byte_list.decode("utf-8", errors="replace")

    def token_to_id(self, token: str) -> int:
        return self.encoder.get(token, 0)

    def get_padding_token_id(self) -> int:
        if config.SpecialTokens.PAD in self.special_tokens:
            return self.encoder[config.SpecialTokens.PAD]
        raise ValueError("Padding token not defined.")

    def strip(self, text: str) -> str:
        pattern = "|".join(re.escape(t) for t in self.special_tokens)
        return re.sub(pattern, "", text).strip()

    def save(self, folder: str, filename_prefix: str = "bpe_tokenizer"):
        os.makedirs(folder, exist_ok=True)

        with open(
            os.path.join(folder, f"vocab_{filename_prefix}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        with open(
            os.path.join(folder, f"merges_{filename_prefix}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("#version: 0.2\n")
            for pair, _ in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\n")

    def load(self, folder: str, filename_prefix: str = "bpe_tokenizer"):
        with open(
            os.path.join(folder, f"vocab_{filename_prefix}.json"),
            "r",
            encoding="utf-8",
        ) as f:
            self.encoder = json.load(f)

        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(
            os.path.join(folder, f"merges_{filename_prefix}.txt"),
            "r",
            encoding="utf-8",
        ) as f:
            lines = f.readlines()

        merges = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append(tuple(parts))

        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.cache = {}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def __repr__(self) -> str:
        return f"ByteLevelBPE(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    tokenizer = ByteLevelBPE()
    print("Created tokenizer:", tokenizer)
