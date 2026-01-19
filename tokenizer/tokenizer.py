import json
import os
import pickle
import re
from typing import Dict, List, Tuple
from collections import Counter
import sys

from tokenizers import Tokenizer
import config

import transformers
import warnings
import torch

from tokenizers.processors import TemplateProcessing

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
        print('special_tokens_dict:', special_tokens_dict)

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

    # max_seq_length <= 0 means no truncation/padding
    # returns a dictionary with 'input_ids' and 'attention_mask' as keys
    def encode(
        self,
        text: str,
        max_seq_length: int,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:

        tokens = self.tokenize(text)
        ids = [self.encoder.get(token, 0) for token in tokens]

        if max_seq_length > 0 and len(ids) > max_seq_length:
            ids = ids[:max_seq_length]
            if verbose:
                print(f"Warning: input text truncated to {max_seq_length} tokens.")
        if config.SpecialTokens.BOS and config.SpecialTokens.EOS in self.special_tokens:
            if max_seq_length > 0 and len(ids) > max_seq_length - 2:
                ids = ids[:-2]
            ids = [self.encoder[config.SpecialTokens.BOS]] + ids + [self.encoder[config.SpecialTokens.EOS]]
            if verbose:
                print(f"Warning: Added BOS and EOS tokens, total length is now {len(ids)}.")
        if max_seq_length > 0 and config.SpecialTokens.PAD in self.special_tokens:
            ids = ids + (max_seq_length - len(ids)) * [self.encoder[config.SpecialTokens.PAD]]
            if verbose:
                print(f"Warning: Added PAD tokens, total length is now {len(ids)}.")

        output = {'input_ids': torch.tensor(ids, dtype=torch.long)}
        if max_seq_length > 0:
            attention_mask = [1 if id != self.encoder.get(config.SpecialTokens.PAD, -1) else 0 for id in ids]
            output['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        else:
            output['attention_mask'] = torch.tensor([1]*len(ids), dtype=torch.long)
        return output
    
    def encode_batched(
        self,
        texts: List[str],
        max_seq_length: int,
        padding: bool = True,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:

        batch_input_ids = []
        batch_attention_mask = []

        for text in texts:
            encoded = self.encode(text, max_seq_length, verbose)
            input_ids = encoded['input_ids'].tolist()
            attention_mask = encoded['attention_mask'].tolist()
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long)
        }
    
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = []

        for token_id in ids:
            token_str = self.decoder.get(token_id, '')
            tokens.append(token_str)

        text = ''.join(tokens)
        byte_list = bytearray()

        for char in text:
            if char in self.byte_decoder:
                byte_list.append(self.byte_decoder[char])
            else:
                byte_list.append(ord('?'))

        return byte_list.decode('utf-8', errors='replace')

    def decode_batched(self, batch_ids) -> List[str]:
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()
        decoded_texts = []
        for ids in batch_ids:
            decoded_texts.append(self.decode(ids))
        return decoded_texts

    # def decode(self, ids: List[int]) -> str:
    #     tokens = []
    #
    #     for token_id in ids:
    #         token_str = self.decoder.get(token_id, "")
    #         if token_str in self.special_tokens:
    #             continue
    #         tokens.append(token_str)
    #
    #     text = "".join(tokens)
    #     byte_list = bytearray()
    #
    #     for char in text:
    #         if char in self.byte_decoder:
    #             byte_list.append(self.byte_decoder[char])
    #         else:
    #             byte_list.append(ord("?"))
    #
    #     return byte_list.decode("utf-8", errors="replace")

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

    def get_vocab_size(self) -> int:
        return len(self.encoder)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.encoder

    def __repr__(self) -> str:
        return f"ByteLevelBPE(vocab_size={self.get_vocab_size()})"
    
class TokenizerHF:

    def __init__(self, tokenizer_name = "gpt2") -> None:
        # https://discuss.huggingface.co/t/gpt2tokenizer-not-putting-bos-eos-token/27394/2
        bos = config.SpecialTokens.BOS.value
        eos = config.SpecialTokens.EOS.value
        pad = config.SpecialTokens.PAD.value
        special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
        print("Initializing HF Tokenizer with special tokens:", special_tokens_dict)

        tokenizer_orig = transformers.AutoTokenizer.from_pretrained(tokenizer_name) # transformer library
        tokenizer_orig.add_special_tokens(special_tokens_dict) # with this, you don't have to manually define the new tokens' ids
        tokenizer = Tokenizer.from_pretrained(tokenizer_name) # tokenizer library
        tokenizer.post_processor = TemplateProcessing(
            single=bos + " $A " + eos,
            special_tokens=[(eos, tokenizer_orig.eos_token_id), (bos, tokenizer_orig.bos_token_id)],
        )
        self.tokenizer = transformers.GPT2TokenizerFast(tokenizer_object=tokenizer) #transformer library again but now with post processing
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.vocab_size = self.tokenizer.vocab_size + 3
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad) #self.tokenizer.convert_tokens_to_ids(config.SpecialTokens.PAD.value)
        print(f"HF Tokenizer initialized with vocab size: {self.vocab_size}, pad_token_id: {self.pad_token_id}")

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    # returns a dictionary with 'input_ids' and 'attention_mask' as keys
    def encode_batched(self, texts: List[str], max_seq_length: int, padding=True, verbose=False) -> Dict[str, torch.Tensor]:
        return self.tokenizer(texts, max_length=max_seq_length, padding='max_length' if padding else True, 
                              return_tensors='pt', truncation=True)
    
    def encode(self, text: str, max_seq_length: int, verbose: bool = False) -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, max_length=max_seq_length, padding='max_length', 
                              return_tensors='pt', truncation=True)
        
    def decode(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def decode_batched(self, token_ids) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
    
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
    
    def strip(self, text: str) -> str:
        pattern = "|".join(re.escape(t) for t in [config.SpecialTokens.PAD, config.SpecialTokens.BOS, config.SpecialTokens.EOS])
        return re.sub(pattern, "", text).strip()

    def get_padding_token_id(self):
        return self.pad_token_id

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def save(self, folder: str, filename_prefix: str = "hf_tokenizer"):
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"vocab_{filename_prefix}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(folder: str, filename_prefix: str = "hf_tokenizer"):
        file_path = os.path.join(folder, f"vocab_{filename_prefix}.pkl")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    def get_vocab_size(self) -> int:
        return self.vocab_size

if __name__ == "__main__":
    if config.TOKENIZER_TYPE == 'hf':
        tokenizer = TokenizerHF(tokenizer_name="gpt2")
        print("Created HF tokenizer:", tokenizer)
    else:
        tokenizer = ByteLevelBPE()
        print("Created tokenizer:", tokenizer)