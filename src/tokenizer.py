"""Turn a text into a list of normalized words."""

import re

import Stemmer

PARTS_PATTERN = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+"
STEMMER = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    """Split a text into lowercase stemmed words."""
    tokens: list[str] = []
    for mot in re.findall(r"\w+", text):
        tokens.append(mot.lower())
        morceaux = re.findall(PARTS_PATTERN, mot)
        if len(morceaux) > 1:
            for morceau in morceaux:
                tokens.append(morceau.lower())
    stemmed: list[str] = STEMMER.stemWords(tokens)
    return stemmed


if __name__ == "__main__":
    print(tokenize("How do I generate embedding vectors with chat templates?"))
