"""Show the questions whose correct sources are missing from the top k."""

import sys

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

K = 5


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
    return union > 0 and inter / union >= 0.05


def main(results_path: str, answers_path: str) -> None:
    """Print every question with no correct source in the top K."""
    with open(results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate_json(f.read())
    with open(answers_path, encoding="utf-8") as f:
        answers = RagDataset.model_validate_json(f.read())
    attendues = {
        q.question_id: q.sources
        for q in answers.rag_questions
        if isinstance(q, AnsweredQuestion)
    }
    ratees = 0
    for result in results.search_results:
        correctes = attendues.get(result.question_id, [])
        top = result.retrieved_sources[:K]
        trouve = any(overlaps(s, c) for s in top for c in correctes)
        if not trouve:
            ratees += 1
            print("QUESTION :", result.question)
            for c in correctes:
                print(
                    "  attendu :", c.file_path,
                    f"[{c.first_character_index}:{c.last_character_index}]",
                )
            for s in top[:3]:
                print("  trouvé  :", s.file_path)
            print()
    print(ratees, "questions ratées sur", len(results.search_results))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])