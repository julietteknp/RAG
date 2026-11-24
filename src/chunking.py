"""Split corpus files into chunks."""

import ast
import warnings

from src.models import Chunk


def make_chunks(
    text: str, file_path: str, start: int, end: int, max_size: int,
    title: str,
) -> list[Chunk]:
    """Split text[start:end] into chunks of at most max_size characters."""
    chunks: list[Chunk] = []
    while start < end:
        limit = start + max_size
        if limit >= end:
            # tout ce qui reste tient dans un seul morceau
            cut = end
        else:
            # 1er choix : couper après une ligne vide (fin de paragraphe)
            cut = text.rfind("\n\n", start, limit)
            if cut > start:
                cut += 2
            else:
                # 2e choix : couper après une fin de ligne
                cut = text.rfind("\n", start, limit)
                if cut > start:
                    cut += 1
                else:
                    # dernier recours : couper pile à la limite
                    cut = limit
        chunks.append(Chunk(
            file_path=file_path,
            first_character_index=start,
            last_character_index=cut,
            search_text=title + "\n" + text[start:cut],
        ))
        start = cut
    return chunks


def chunk_markdown(text: str, file_path: str, max_size: int) -> list[Chunk]:
    """Split a Markdown text into chunks, one section per heading."""
    chunks: list[Chunk] = []
    start = 0
    title = ""
    i = 0
    while i < len(text):
        debut_de_ligne = i == 0 or text[i - 1] == "\n"
        if text[i] == "#" and debut_de_ligne:
            # 1. créer les chunks de la section précédente (si non vide)
            if i > start:
                chunks.extend(
                    make_chunks(text, file_path, start, i, max_size, title)
                )
            # 2. la nouvelle section commence ici
            start = i
            # 3. lire le titre de la nouvelle section
            fin_de_ligne = text.find("\n", i)
            if fin_de_ligne == -1:
                fin_de_ligne = len(text)
            title = text[i:fin_de_ligne].strip("#").strip()
        i += 1
    # la dernière section, qui va jusqu'à la fin du texte
    if start < len(text):
        chunks.extend(
            make_chunks(text, file_path, start, len(text), max_size, title)
        )
    return chunks


def chunk_python(text: str, file_path: str, max_size: int) -> list[Chunk]:
    """Split Python code into chunks, one block per top-level def or class."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return make_chunks(text, file_path, 0, len(text), max_size, "")
    line_starts = [0]
    for i, char in enumerate(text):
        if char == "\n":
            line_starts.append(i + 1)
    boundaries: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            first_line = min(
                [node.lineno] + [d.lineno for d in node.decorator_list]
            )
            boundaries.append((line_starts[first_line - 1], node.name))
    chunks: list[Chunk] = []
    start = 0
    title = ""
    for offset, name in boundaries:
        if offset > start:
            chunks.extend(
                make_chunks(text, file_path, start, offset, max_size, title)
            )
        start = offset
        title = name
    if start < len(text):
        chunks.extend(
            make_chunks(text, file_path, start, len(text), max_size, title)
        )
    return chunks


def chunk_file(text: str, file_path: str, max_size: int) -> list[Chunk]:
    """Split a file into chunks, depending on its extension."""
    if file_path.endswith(".md"):
        return chunk_markdown(text, file_path, max_size)
    if file_path.endswith(".py"):
        return chunk_python(text, file_path, max_size)
    return make_chunks(text, file_path, 0, len(text), max_size, "")


if __name__ == "__main__":
    code = (
        "import os\n\nX = 1\n\n\n@cache\ndef load_config():\n"
        "    return X\n\n\nclass Server:\n    def start(self):\n"
        "        pass\n"
    )
    for chunk in chunk_python(code, "test.py", 2000):
        print(chunk.first_character_index, chunk.last_character_index)
        print(repr(chunk.search_text))
