import re

PARTS_PATTERN = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+"


def tokenize(text: str) -> list[str]:
    """Split a text into lowercase words, also splitting code identifiers."""
    tokens: list[str] = []
    for mot in re.findall(r"\w+", text):
        tokens.append(mot.lower())
        morceaux = re.findall(PARTS_PATTERN, mot)
        if len(morceaux) > 1:
            for morceau in morceaux:
                tokens.append(morceau.lower())
    return tokens
