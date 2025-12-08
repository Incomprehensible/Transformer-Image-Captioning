from tokenizer import ByteLevelBPE

sample_corpus = [
    "A dog is playing in the snow.",
    "A cat is looking at the window."
    "A bird is flying."
]

print("Running tests")

tok = ByteLevelBPE()
tok.train(sample_corpus, vocab_size=295, verbose=True)

text = "A dog is playing in the snow."
ids = tok.encode(text)
decoded = tok.decode(ids)

print("\nOriginal text:", text)
print("Encoded IDs:", ids)
print("Decoded text:", decoded)

assert decoded == text, "Decoded text does NOT match original!"

save_folder = "test_tokenizer_save"
tok.save(save_folder)

tok2 = ByteLevelBPE()
tok2.load(save_folder)

ids2 = tok2.encode(text)
decoded2 = tok2.decode(ids2)

print("\nAfter reload - Encoded IDs:", ids2)
print("After reload - Decoded text:", decoded2)

assert decoded2 == text, "Loaded tokenizer mismatch!"

import shutil
shutil.rmtree(save_folder, ignore_errors=True)

print("\nALL TESTS PASSED")
