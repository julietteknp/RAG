"""Compute the recall@k of search results against a ground-truth dataset."""

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

MIN_IOU = 0.05


def overlaps(a: MinimalSource, b: MinimalSource) -> bool:
    """Return True if a and b are in the same file and overlap enough."""
    if a.file_path != b.file_path:
        return False
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    union = max(a.last_character_index, b.last_character_index) - min(
        a.first_character_index, b.first_character_index
    )
    return union > 0 and inter / union >= MIN_IOU


def question_recall(
    retrieved: list[MinimalSource], correct: list[MinimalSource], k: int
) -> float:
    """Return the share of correct sources found in the top k retrieved."""
    top = retrieved[:k]
    found = 0
    for source in correct:
        if any(overlaps(r, source) for r in top):
            found += 1
    return found / len(correct)


def run_evaluate(results_path: str, dataset_path: str) -> None:
    """Print recall@1, 3, 5 and 10 of the results against the dataset."""
    with open(results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate_json(f.read())
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    expected = {
        q.question_id: q.sources
        for q in dataset.rag_questions
        if isinstance(q, AnsweredQuestion) and q.sources
    }
    retrieved = {
        r.question_id: r.retrieved_sources for r in results.search_results
    }
    if not expected:
        print("No question with sources to evaluate")
        return
    print(f"Questions evaluated: {len(expected)}")
    for k in (1, 3, 5, 10):
        recalls = [
            question_recall(retrieved.get(question_id, []), sources, k)
            for question_id, sources in expected.items()
        ]
        recall = sum(recalls) / len(recalls)
        print(f"Recall@{k}: {recall:.3f} ({recall:.1%})")
