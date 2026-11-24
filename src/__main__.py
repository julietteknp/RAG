"""Command-line interface of the RAG system."""

import fire

from src.evaluate import run_evaluate
from src.generator import Generator, run_answer_dataset
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
            print(
                "Error: max_chunk_size must be an integer between 1 and 2000"
            )
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

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print the recall@k of search results against a dataset."""
        try:
            run_evaluate(str(student_search_results_path), str(dataset_path))
        except Exception as error:
            print(f"Error: {error}")

    def answer(
        self, query: str, k: int = 5, processed_dir: str = "data/processed"
    ) -> None:
        """Answer a single query using the retrieved context."""
        if not isinstance(k, int):
            print("Error: k must be an integer")
            return
        if not str(query).strip():
            print("Error: the query is empty")
            return
        try:
            sources = Retriever(processed_dir).search(str(query), k)
            answer = Generator().answer(str(query), sources)
        except Exception as error:
            print(f"Error: {error}")
            return
        print("Sources:")
        for source in sources:
            print(
                f"  {source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )
        print("Answer:")
        print(answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for every question of a search results file."""
        try:
            run_answer_dataset(
                str(student_search_results_path), str(save_directory)
            )
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    fire.Fire(RagCLI)
