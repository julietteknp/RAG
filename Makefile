UNANSWERED = data/datasets/UnansweredQuestions
ANSWERED = data/datasets/AnsweredQuestions
SEARCH_OUT = data/output/search_results/UnansweredQuestions
ANSWER_OUT = data/output/search_results_and_answer/UnansweredQuestions
K ?= 10
ARGS ?= index

FLAKE8_EXCLUDE = .venv,data
MYPY_EXCLUDE = '^(\.venv|data)/'

.PHONY: install run index search evaluate answer debug clean lint lint-strict

install:
	uv sync

run: index search evaluate

index:
	uv run python -m src index

search:
	uv run python -m src search_dataset \
		--dataset_path $(UNANSWERED)/dataset_docs_public.json \
		--k $(K) --save_directory $(SEARCH_OUT)
	uv run python -m src search_dataset \
		--dataset_path $(UNANSWERED)/dataset_code_public.json \
		--k $(K) --save_directory $(SEARCH_OUT)

evaluate:
	uv run python -m src evaluate \
		--student_search_results_path $(SEARCH_OUT)/dataset_docs_public.json \
		--dataset_path $(ANSWERED)/dataset_docs_public.json
	uv run python -m src evaluate \
		--student_search_results_path $(SEARCH_OUT)/dataset_code_public.json \
		--dataset_path $(ANSWERED)/dataset_code_public.json

answer:
	uv run python -m src answer_dataset \
		--student_search_results_path $(SEARCH_OUT)/dataset_docs_public.json \
		--save_directory $(ANSWER_OUT)
	uv run python -m src answer_dataset \
		--student_search_results_path $(SEARCH_OUT)/dataset_code_public.json \
		--save_directory $(ANSWER_OUT)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 . --exclude $(FLAKE8_EXCLUDE)
	uv run mypy . --exclude $(MYPY_EXCLUDE) \
		--warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude $(FLAKE8_EXCLUDE)
	uv run mypy . --exclude $(MYPY_EXCLUDE) --strict