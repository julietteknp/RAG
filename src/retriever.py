"""Search the BM25 index for the chunks most relevant to a question."""

import json
from pathlib import Path

import bm25s
from tqdm import tqdm

from src.models import (
    Chunk,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.tokenizer import tokenize


class Retriever:
    """Load a saved index and search it."""

    def __init__(self, index_dir: str) -> None:
        """Load the BM25 index and the chunks from index_dir."""
        self.bm25 = bm25s.BM25.load(str(Path(index_dir) / "bm25"))
        with open(Path(index_dir) / "chunks.json", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [Chunk.model_validate(item) for item in data]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the k sources most relevant to the query."""
        query_tokens = tokenize(query)
        if not query_tokens or k <= 0:
            return []
        k = min(k, len(self.chunks))
        results, scores = self.bm25.retrieve([query_tokens], k=k)
        sources: list[MinimalSource] = []
        for index in results[0]:
            chunk = self.chunks[index]
            sources.append(MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            ))
        return sources


def run_search_dataset(
    dataset_path: str, k: int, save_directory: str, processed_dir: str
) -> None:
    """Search every question of a dataset and save the results as JSON."""
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    retriever = Retriever(processed_dir)
    results: list[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        sources = retriever.search(question.question, k)
        results.append(MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        ))
    output = StudentSearchResults(search_results=results, k=k)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_directory) / Path(dataset_path).name
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    retriever = Retriever("data/processed")
    tests = [
        ("How do I enable LoRA?", 5),
        ("", 5),
        ("???", 5),
        ("zzzz qqqq", 5),
        ("How do I enable LoRA?", 0),
        ("How do I enable LoRA?", 100000),
    ]
    for query, k in tests:
        sources = retriever.search(query, k)
        print(repr(query), "k =", k, "->", len(sources), "résultats")