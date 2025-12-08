from typing import Dict, Tuple, Set

def byte_to_unicode() -> Dict[int, str]:
    bs = []
    bs.extend(range(ord("!"), ord("~") + 1))
    bs.extend(range(ord("¡"), ord("¬") + 1))
    bs.extend(range(ord("®"), ord("ÿ") + 1))
    cs = bs.copy()
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs_chars = [chr(c) for c in cs]

    return dict(zip(bs, cs_chars))

def unicode_to_byte_map() -> Dict[str, int]:
    byte_to_uni = byte_to_unicode()

    return {char: byte_val for byte_val, char in byte_to_uni.items()}


def get_pairs(word: tuple) -> Set[Tuple[str, str]]:
    pairs = set()

    if len(word) < 2:
        return pairs

    prev = word[0]

    for current in word[1:]:
        pairs.add((prev, current))
        prev = current

    return pairs


def get_stats(vocab: Dict[tuple, int]) -> Dict[Tuple[str, str], int]:
    pair_counts = {}

    for word_tuple, freq in vocab.items():
        word_pairs = get_pairs(word_tuple)

        for pair in word_pairs:
            if pair in pair_counts:
                pair_counts[pair] += freq
            else:
                pair_counts[pair] = freq

    return pair_counts


def merge_vocab(pair: Tuple[str, str], vocab: Dict[tuple, int]) -> Dict[tuple, int]:
    new_vocab = {}
    first, second = pair
    merged_symbol = first + second
    pair_str = f"{first} {second}"
    merged_str = merged_symbol

    for word_tuple, freq in vocab.items():
        word_str = ' '.join(word_tuple)
        new_word_str = word_str.replace(pair_str, merged_str)
        new_word_tuple = tuple(new_word_str.split())
        new_vocab[new_word_tuple] = freq

    return new_vocab


if __name__ == "__main__":
    b2u = byte_to_unicode()
    u2b = unicode_to_byte_map()
    print(f"Mapped {len(b2u)} bytes to Unicode")

    for b in range(256):
        if b not in b2u:
            print(f"ERROR: byte {b} not in mapping!")
            break
        char = b2u[b]
        if char not in u2b:
            print(f"ERROR: char '{char}' not in reverse mapping!")
            break
        if u2b[char] != b:
            print(f"ERROR: round trip failed for byte {b}")
            break
    else:
        print("All bytes round-trip correctly")

    test_word = ('l', 'o', 'w', 'e', 'r')
    pairs = get_pairs(test_word)
    print(f"Word {test_word} -> pairs: {pairs}")
    assert len(pairs) == 4, f"Expected 4 pairs, got {len(pairs)}"

    test_vocab = {
        ('l', 'o', 'w'): 5,
        ('l', 'o', 'w', 'e', 'r'): 3,
        ('n', 'e', 'w', 'e', 'r'): 2,
    }

    stats = get_stats(test_vocab)
    print(f"Pair frequencies: {stats}")
    most_freq = max(stats, key=stats.get)
    print(f"Most frequent pair: {most_freq}")
    merged = merge_vocab(most_freq, test_vocab)
    print(f"After merging: {merged}")
    print("\nAll tests passed!")
