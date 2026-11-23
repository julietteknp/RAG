"""Build the search index from the corpus."""

from pathlib import Path

import bm25s
from tqdm import tqdm
import json

from src.chunking import chunk_file
from src.models import Chunk
from src.tokenizer import tokenize


def list_files(raw_dir: str) -> list[str]:
    """Return the paths of all .py, .md and .txt files under raw_dir."""
    files: list[str] = []
    for path in Path(raw_dir).rglob("*"):
        if path.is_file() and path.suffix in (".py", ".md", ".txt"):
            files.append(str(path))
    return sorted(files)


def build_chunks(files: list[str], max_size: int) -> list[Chunk]:
    """Read every file and split it into chunks."""
    chunks: list[Chunk] = []
    for file_path in tqdm(files, desc="Chunking"):
        try:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        chunks.extend(chunk_file(text, file_path, max_size))
    return chunks


def build_index(chunks: list[Chunk]) -> bm25s.BM25:
    """Build a BM25 index from the chunks."""
    corpus_tokens: list[list[str]] = []
    for chunk in tqdm(chunks, desc="Tokenizing"):
        text = chunk.file_path + " " + chunk.search_text
        corpus_tokens.append(tokenize(text))
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    return bm25

def save_index(bm25: bm25s.BM25, chunks: list[Chunk], out_dir: str) -> None:
    """Save the BM25 index and the chunks under out_dir."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    bm25.save(str(Path(out_dir) / "bm25"))
    with open(Path(out_dir) / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump() for chunk in chunks], f)

def run_index(raw_dir: str, out_dir: str, max_size: int) -> None:
    """Chunk the corpus, build the BM25 index and save it."""
    files = list_files(raw_dir)
    chunks = build_chunks(files, max_size)
    bm25 = build_index(chunks)
    save_index(bm25, chunks, out_dir)
    print(f"Ingestion complete! Indices saved under {out_dir}")

if __name__ == "__main__":
    run_index("data/raw", "data/processed", 2000)