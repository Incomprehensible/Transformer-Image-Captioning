from tokenizer import ByteLevelBPE
from transformers import AutoTokenizer

sample_corpus = [
    "A dog is playing in the snow.",
    "A cat is looking at the window.",
]

test_texts = [
    "A dog is playing in the snow.",
    "A cat is looking at the window.",
]

print("Running tokenizer comparison tests")

print("\n=== Our ByteLevelBPE tests ===")

tok = ByteLevelBPE()
tok.train(sample_corpus, vocab_size=295, verbose=True)

print("\n[Our]")

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)

    print("Original:", text)
    print("IDs:", ids)
    print("Decoded:", decoded)

    assert decoded == text, "Our tokenizer decode mismatch!"

print("Our tokenizer passed all tests.")

hf_models = {
    "gpt2": "gpt2",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
}

hf_tokenizers = {
    name: AutoTokenizer.from_pretrained(model)
    for name, model in hf_models.items()
}

print("\n=== Hugging Face tokenizers tests ===")

for name, tokenizer in hf_tokenizers.items():
    print(f"\n[HF: {name}]")

    for text in test_texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )

        ids = encoded["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        print("Original:", text)
        print("IDs:", ids)
        print("Decoded:", decoded)

        assert isinstance(ids, list)
        assert len(ids) > 0
        assert isinstance(decoded, str)

        if name in {"gpt2", "roberta"}:
            assert decoded == text, f"{name} failed round-trip!"

    print(f"HF tokenizer '{name}' passed all tests")

print("\nAll tokenizers passed all tests.")
