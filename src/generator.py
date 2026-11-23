"""Generate an answer to a question from retrieved sources with Qwen."""

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import Retriever

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_TOKENS = 3000
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM project. "
    "Answer using only the provided context. "
    "If the context does not contain the answer, say that you do not know. "
    "Be concise and precise."
)


def read_source(source: MinimalSource) -> str:
    """Return the text of the corpus covered by a source."""
    try:
        with open(source.file_path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return ""
    return text[source.first_character_index:source.last_character_index]


def pick_device() -> str:
    """Return the fastest available device: cuda, mps or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Generator:
    """Load Qwen and generate answers grounded in retrieved sources."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and the model on the best available device."""
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(
            self.device
        )
        print(f"Model loaded on {self.device}")

    def count_tokens(self, text: str) -> int:
        """Return the number of model tokens in a text."""
        return len(self.tokenizer(text)["input_ids"])

    def build_context(self, sources: list[MinimalSource]) -> str:
        """Join the source texts without exceeding the token budget."""
        parts: list[str] = []
        used = 0
        for number, source in enumerate(sources, start=1):
            text = read_source(source)
            if not text:
                continue
            part = f"[{number}] {source.file_path}\n{text}"
            tokens = self.count_tokens(part)
            if used + tokens > MAX_CONTEXT_TOKENS:
                break
            parts.append(part)
            used += tokens
        return "\n\n".join(parts)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate an answer to the question from the sources."""
        context = self.build_context(sources)
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        prompt_length = inputs["input_ids"].shape[1]
        answer: str = self.tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )
        return answer.strip()


def run_answer_dataset(results_path: str, save_directory: str) -> None:
    """Answer every question of a search results file and save them."""
    with open(results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate_json(f.read())
    print(f"Loaded {len(results.search_results)} questions")
    generator = Generator()
    answers: list[MinimalAnswer] = []
    for result in tqdm(results.search_results, desc="Answering"):
        answer = generator.answer(result.question, result.retrieved_sources)
        answers.append(MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=answer,
        ))
    output = StudentSearchResultsAndAnswer(search_results=answers, k=results.k)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_directory) / Path(results_path).name
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    print(f"Saved student_search_results_and_answer to {save_path}")


if __name__ == "__main__":
    question = "How do I enable LoRA in vLLM?"
    sources = Retriever("data/processed").search(question, 5)
    generator = Generator()
    context = generator.build_context(sources)
    print("SOURCES DANS LE CONTEXTE :")
    for line in context.splitlines():
        if line.startswith("[") and "data/raw/" in line:
            print(" ", line)
    print("TOKENS DU CONTEXTE :", generator.count_tokens(context))
    print("RÉPONSE :")
    print(generator.answer(question, sources))