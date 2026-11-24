from src.models import RagDataset

with open("data/datasets/AnsweredQuestions/dataset_docs_public.json") as f:
    dataset = RagDataset.model_validate_json(f.read())

print(len(dataset.rag_questions), "questions chargées")
print(dataset.rag_questions[0])
