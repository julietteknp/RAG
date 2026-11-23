"""Command-line interface of the RAG system."""

import fire

from src.indexer import run_index
from src.retriever import Retriever, run_search_dataset


class RagCLI:
    """Commands of the RAG system."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Chunk the corpus and build the BM25 index."""
        if not isinstance(max_chunk_size, int) or not (
            0 < max_chunk_size <= 2000
        ):
            print("Error: max_chunk_size must be an integer between 1 and 2000")
            return
        try:
            run_index(raw_dir, processed_dir, max_chunk_size)
        except Exception as error:
            print(f"Error: {error}")

    def search(
        self, query: str, k: int = 10, processed_dir: str = "data/processed"
    ) -> None:
        """Print the top-k sources for a single query."""
        if not isinstance(k, int):
            print("Error: k must be an integer")
            return
        try:
            retriever = Retriever(processed_dir)
            sources = retriever.search(str(query), k)
        except Exception as error:
            print(f"Error: {error}")
            return
        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )
    
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
    ) -> None:
        """Search every question of a dataset and save the results."""
        if not isinstance(k, int) or k < 0:
            print("Error: k must be a non-negative integer")
            return
        try:
            run_search_dataset(
                str(dataset_path), k, str(save_directory), processed_dir
            )
        except Exception as error:
            print(f"Error: {error}")

if __name__ == "__main__":
    fire.Fire(RagCLI)