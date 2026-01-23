from statistics import mean
from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


class BLUE:
    def __init__(self, ngrams: int = 4) -> None:
        self.smoothing = SmoothingFunction().method3
        self.n = ngrams
        weights = [1 / ngrams if i <= ngrams else 0 for i in range(1, 5)]
        self.weights = tuple(weights)

    def __call__(self, references, hypothesis) -> float:
        return sentence_bleu(
            references,
            hypothesis,
            weights=self.weights,
            smoothing_function=self.smoothing
        )

    def __repr__(self) -> str:
        return f"bleu{self.n}"


class GLEU:
    def __call__(self, *args, **kwargs):
        return sentence_gleu(*args, **kwargs)

    def __repr__(self):
        return "gleu"


class METEOR:
    def __call__(self, *args, **kwargs):
        return meteor_score(*args, **kwargs)

    def __repr__(self):
        return "meteor"


class Metrics:
    def __init__(self) -> None:
        self.bleu1 = BLUE(ngrams=1)
        self.bleu2 = BLUE(ngrams=2)
        self.bleu3 = BLUE(ngrams=3)
        self.bleu4 = BLUE(ngrams=4)

        self.gleu = GLEU()
        self.meteor = METEOR()

        self.all = [self.bleu1, self.bleu2, self.bleu3, self.bleu4, self.gleu, self.meteor]

    # The expected type for hypothesis is list(str)
    # candidate is a list(list(str))
    def calculate(
        self,
        refs: List[List[List[str]]],
        hypos: List[List[str]],
        train: bool = False
    ) -> Dict[str, float]:

        score_fns = [self.bleu4, self.meteor] if train else self.all

        return {
            repr(fn): mean(map(fn, refs, hypos))
            for fn in score_fns
        }
