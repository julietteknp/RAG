*This activity has been created as part of the 42 curriculum by jkonop*
# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) codebase (version 0.10.1). It
indexes the repository, retrieves the most relevant code and documentation
snippets for a question with BM25, and asks a small local language model
(`Qwen/Qwen3-0.6B`) to write an answer grounded in those snippets.

---

## Description

A language model only knows what it saw during training. It cannot answer
reliably about a private codebase or a specific version of a library, and it
tends to invent an answer instead. Retraining it every time the data changes
is far too expensive.

RAG solves this by giving the model the right documents at answer time, like
an open-book exam: a search engine finds the few relevant passages, and the
model reads them before answering.

This project implements the four classic stages of RAG:

1. **Indexing**: every `.py`, `.md` and `.txt` file of the vLLM repository is
   split into chunks of at most 2000 characters, and a BM25 index is built and
   saved to disk.
2. **Retrieving**: a question is tokenized the same way as the chunks, and
   BM25 returns the `k` best chunks, each identified by its file path and
   character range.
3. **Augmenting**: the text of the best chunks is put into the model's
   context, within a fixed token budget.
4. **Generating**: `Qwen/Qwen3-0.6B` reads this context and writes the answer.

Retrieval quality is measured with **recall@k**. On the public datasets the
system reaches **85.0 % recall@5 on documentation questions** and **82.8 %
recall@5 on code questions** (required: 80 % and 50 %).

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (project and package manager)
- A few GB of free disk space (PyTorch and the model weights)

### Installation

```bash
make install        # runs `uv sync`
```

The data is not part of the repository. Place the files provided with the
subject as follows:

```
data/raw/vllm-0.10.1/                 # the vLLM repository to index
data/datasets/UnansweredQuestions/    # questions (input of search_dataset)
data/datasets/AnsweredQuestions/      # ground truth (for evaluate)
```

The first time an answer is generated, the model is downloaded automatically
from Hugging Face (about 1.5 GB) and cached in `~/.cache/huggingface/`.

### Running

Every command is run as `uv run python -m src <command> [options]`.

| Command | Description |
|---|---|
| `index --max_chunk_size 2000` | Chunk `data/raw/` and save the index under `data/processed/` |
| `search "<query>" --k 5` | Print the top-k sources for one question |
| `search_dataset --dataset_path <file> --k 10 --save_directory <dir>` | Search every question of a dataset and save a `StudentSearchResults` JSON file |
| `answer "<query>" --k 5` | Retrieve sources for one question and print the generated answer |
| `answer_dataset --student_search_results_path <file> --save_directory <dir>` | Generate an answer for every question of a search results file and save a `StudentSearchResultsAndAnswer` JSON file |
| `evaluate --student_search_results_path <file> --dataset_path <file>` | Print recall@1, 3, 5 and 10 against a ground-truth dataset |

Every input and output path is a command-line option with a default value; no
path is hard-coded.

### Makefile

| Rule | Description |
|---|---|
| `make install` | Install the dependencies (`uv sync`) |
| `make run` | Full retrieval pipeline: `index`, then `search`, then `evaluate` |
| `make index` | Build the index |
| `make search` | Run `search_dataset` on the docs and code public datasets |
| `make evaluate` | Print the recall of both datasets |
| `make answer` | Run `answer_dataset` on both datasets (slow) |
| `make debug` | Run the program under `pdb` (`make debug ARGS='search "LoRA"'`) |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` | Run `flake8` and `mypy` with the flags required by the subject |
| `make lint-strict` | Run `flake8` and `mypy --strict` |

`K` can be overridden: `make run K=5`.

---

## System architecture

```
src/
├── __main__.py    Command-line interface (Python Fire): the six commands
├── models.py      Pydantic models exchanged between the stages
├── tokenizer.py   Text -> list of normalized, stemmed words (for BM25)
├── chunking.py    File -> list of chunks (Markdown, Python and text strategies)
├── indexer.py     Stage 1: list files, chunk, build and save the BM25 index
├── retriever.py   Stage 2: load the index, return the top-k sources
├── generator.py   Stages 3 and 4: build the context, generate with Qwen
└── evaluate.py    Recall@k computation
scripts/
└── analyze.py     Development tool: list the questions missed in the top k
```

### Data flow

```
data/raw/  ──index──▶  data/processed/ (bm25/ + chunks.json)
                              │
questions.json ──search_dataset──▶ search results JSON ──evaluate──▶ recall@k
                                          │
                                   answer_dataset
                                          │
                                          ▼
                           search results + answers JSON
```

### Components

- **`models.py`** contains the pydantic models of the subject
  (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`,
  `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`,
  `StudentSearchResultsAndAnswer`) and one additional model, `Chunk`. A
  `Chunk` inherits from `MinimalSource` (file path, start and end positions)
  and adds a `search_text` field, the text that BM25 reads. Because a chunk
  *is* a source, writing the output only requires copying its three location
  fields.
- **`indexer.py`** runs the indexing stage: `list_files` finds the files,
  `build_chunks` reads and chunks them, `build_index` tokenizes each chunk and
  builds the BM25 index, `save_index` writes the index and the chunk list to
  disk. `run_index` chains the four steps.
- **`retriever.py`** contains the `Retriever` class. It loads the index and
  the chunks **once**, then answers any number of queries. BM25 returns chunk
  *numbers*; the retriever maps them back to `chunks[i]`, which works because
  the chunk list is saved in the same order as the BM25 corpus.
  `run_search_dataset` processes a whole dataset.
- **`generator.py`** contains the `Generator` class. It loads Qwen **once**,
  re-reads the text of each source from the corpus (`file[start:end]`),
  assembles the context within the token budget, builds the chat prompt and
  generates the answer. `run_answer_dataset` processes a whole search results
  file without running the search again.
- **`__main__.py`** only validates the user input, calls the functions above
  and turns every error into a short message: no command ever ends with a
  traceback.

---

## Chunking strategy

Every chunk is at most `--max_chunk_size` characters (2000 by default, the
maximum accepted by the grader). For each chunk the system stores the exact
character range in the original file, so that `file[start:end]` gives back the
chunk text.

### Markdown

Markdown files are split **at every heading** (a line starting with `#`).
Each section (from one heading to the next) becomes one chunk. Text placed
before the first heading forms a chunk of its own.

The heading of the section is **prepended to the `search_text` of every chunk
of that section**. When a long section is split into several chunks, only the
first one contains the heading line in the file; without this, the other
chunks would lose the information of what they are about.

### Python

Python files are parsed with the standard `ast` module, which reads the code
without executing it. The file is split **at every top-level function, async
function and class**: each block runs from one `def`/`class` to the next one.
The decorators written above a definition stay in the same block. The code
placed before the first definition (module docstring, imports, constants)
forms a block of its own.

As for Markdown headings, the **name of the function or class is prepended to
the `search_text` of every chunk of the block**. A large class split into
several chunks keeps its name on each of them.

`ast` gives line numbers, so a table of line start offsets (built by scanning
the text for `\n`) converts them into character positions. If a file cannot be
parsed (`SyntaxError`), it falls back to the generic splitter below, so it is
still indexed.

### Long blocks and text files

Blocks longer than `max_size` (long Markdown sections, large classes) and
`.txt` files are split by a size-bounded splitter (`make_chunks`). Starting
from the beginning, it fills a chunk as much as possible without exceeding
`max_size`, and chooses where to cut in this order of preference:

1. right after a **blank line** (end of a paragraph, or between two methods);
2. right after the **end of a line**;
3. exactly at the limit, only if the chunk contains no line break at all.

The rest of the text goes into the next chunk, so no text is ever lost or
duplicated.

### What is indexed

- **Extensions**: `.py`, `.md` and `.txt`. `.txt` was added after analysing
  the missed questions: three documentation answers were in `CMakeLists.txt`.
  The subject leaves the choice of useful files to the student and defines
  the second strategy as "Markdown / text chunking".
- **Excluded**: any file located in a `tests/` directory (see *Design
  decisions*).

---

## Retrieval method

The system uses **BM25** (library [`bm25s`](https://github.com/xhluca/bm25s),
default parameters).

### Tokenization

BM25 only matches identical words, so the quality of the tokenization is
essential. The same `tokenize` function is applied to chunks and to queries:

1. extract every word made of letters, digits and underscores (`\w+`);
2. keep the **whole word** in lowercase (`run_server`, `openaiserver`), so a
   question quoting an exact identifier still matches;
3. if the word is a code identifier, also add its **parts**: `snake_case` and
   `CamelCase` are split (`run_server` gives `run` and `server`,
   `OpenAIServer` gives `open`, `ai` and `server`), so a question written in
   natural language can match code;
4. apply **Snowball stemming** (`PyStemmer`) so that variants of a word share
   the same root (`embedding`, `embeds` give `embed`; `templates` gives
   `templat`).

### Indexed text

For each chunk, BM25 indexes the **file path followed by the `search_text`**.
The path often contains the most meaningful words of a chunk
(`docs/features/lora.md`, `entrypoints/openai/api_server.py`), even when the
chunk itself does not repeat them. The path is not stored a second time: it is
read from the `file_path` field at indexing time.

### Ranking

BM25 scores every chunk for the query. A chunk earns more points when it
contains query words that are **rare in the corpus** (inverse document
frequency) and when it contains them **several times** (with a saturating
bonus, normalized by chunk length). The `k` best chunks are returned in
decreasing order of score.

### Robustness

- An empty query or a query without any word (`""`, `"???"`) returns no
  source.
- `k <= 0` returns no source; `k` larger than the number of chunks is capped.
- `k`, `max_chunk_size` and the query are validated in the CLI, because Fire
  converts arguments automatically (`--k abc` becomes a string, `search 42`
  becomes an integer).

---

## Performance analysis

### Recall@k on the public datasets

Each line adds one change to the previous one. Measured with the provided
grader (`max_context_length 2000`) and with the `evaluate` command, which
gives identical figures.

**Documentation questions (100)**

| Version | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Baseline: `.py` + `.md`, BM25, identifier splitting | 67.0 % | 77.0 % | 80.0 % | 84.0 % |
| + index `.txt` files | 67.0 % | 78.0 % | 83.0 % | 87.0 % |
| + stemming | 63.0 % | 80.0 % | 86.0 % | 90.0 % |
| + exclude `tests/` | 61.0 % | 80.0 % | 86.0 % | 91.0 % |
| + AST chunking of Python files (final) | 62.0 % | 83.0 % | **85.0 %** | 89.0 % |

**Code questions (99)**

| Version | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Baseline | 51.5 % | 69.7 % | 79.8 % | 86.9 % |
| + index `.txt` files | 51.5 % | 70.7 % | 79.8 % | 86.9 % |
| + stemming | 53.5 % | 73.7 % | 80.8 % | 85.9 % |
| + exclude `tests/` | 52.5 % | 75.8 % | 80.8 % | 87.9 % |
| + AST chunking of Python files (final) | 58.6 % | 79.8 % | **82.8 %** | 88.9 % |

Both required thresholds (80 % docs, 50 % code at recall@5) are met with a
margin.

Splitting Python files at function and class boundaries mostly helps code
questions, especially recall@1 (+6 points): a chunk now holds one whole
function together with its name. On documentation questions it moves recall@5
by one question (86 % to 85 %): the more precise code chunks compete slightly
more with documentation chunks.

Stemming slightly lowers recall@1 on documentation questions: by merging word
variants, more chunks become close to each question, so the right chunk faces
more competition for the first place, but it appears more often in the top 5.
As the target metric is recall@5, the trade-off is positive.

### Analysis of the missed questions

`scripts/analyze.py` lists the questions whose correct source is not in the
top 5. On the baseline, the 20 missed documentation questions fell into three
groups:

| Group | Example | Fix |
|---|---|---|
| Answer in a file that was not indexed | `CMakeLists.txt` (3 questions) | index `.txt` files |
| Right file, wrong part of the file | long tables of `supported_models.md` | heading added to every chunk of a section |
| Different word forms between question and text | "embedding" vs "embed" | stemming |

### Speed

| Step | Measure | Limit |
|---|---|---|
| Indexing the whole corpus | a few seconds | 5 minutes |
| Searching 100 questions | under 1 second | 90 s for 200 questions |
| Generating one answer | about 6 s (Apple M1, MPS) | none |
| `answer_dataset` on 100 questions | about 10 minutes (Apple M1, MPS) | none |

### Answer quality

Most generated answers are coherent, grounded in the retrieved sources and on
point (for example `/v1/load_lora_adapter`, `LLM.embed`,
`SupportsMultiModal`, `--reasoning-parser`). The remaining errors mostly come
from questions where retrieval missed the right source: the model then guesses
or answers off-topic. A few answers are too cautious ("I do not know" although
the information is in the context) or cut by the token limit after copying a
long code example.

---

## Design decisions

- **BM25 rather than TF-IDF.** BM25 adds term-frequency saturation and
  length normalization to TF-IDF, which suits chunks of very different
  lengths. `bm25s` is fast and saves the index to disk.
- **One shared tokenizer for chunks and queries.** Any normalization applied
  on one side must be applied on the other, otherwise words never match.
- **Keep whole identifiers *and* their parts.** Questions sometimes quote an
  exact name (`get_supported_mm_limits`) and sometimes describe it in plain
  words.
- **Index the file path.** It carries strong topic words without increasing
  the stored data.
- **`ast` rather than text patterns for Python.** Searching for `def` in the
  text would also match comments, strings and nested functions; `ast` gives
  the real top-level structure, and handles decorators and `async def`.
- **The same block logic for Markdown and Python.** Both strategies find
  boundaries (headings or definitions), give each block a title, and delegate
  long blocks to the same size-bounded splitter, which keeps the code short
  and the behaviour consistent.
- **Exclude `tests/`.** No public ground-truth source is located in a `tests/`
  directory (checked on both datasets). Test files repeat identifiers such as
  `enable_lora=True` dozens of times, so BM25 ranked them above the
  documentation, and they filled Qwen's context with code that uses a feature
  without explaining it. Excluding them kept recall@5 identical, increased
  recall@10, and fixed answers such as "How do I enable LoRA?".
- **`Chunk` inherits from `MinimalSource`.** The location of a chunk is
  exactly a source, so no conversion logic is needed.
- **Load once, query many times.** `Retriever` and `Generator` are classes
  so that the index and the model are loaded a single time per command.
- **`answer_dataset` reuses the search results file** instead of searching
  again, as in the workflow described by the subject. Retrieval and
  generation can be evaluated and rerun independently.
- **Token budget of 3000 tokens for the context.** Qwen3-0.6B accepts a much
  longer context, but a longer prompt is slower, and a small model gets lost
  in too much text. One chunk is about 500 to 700 tokens, so the budget keeps
  the 4 or 5 best sources. Sources are added in rank order until the next one
  would exceed the budget.
- **Prompt strategy.** A system message asks the model to answer only from
  the provided context, to say when the context does not contain the answer,
  and to be concise. The context lists the sources numbered and labelled with
  their file path, and the question is placed after the context.
- **Thinking mode disabled** (`enable_thinking=False`): Qwen3's reasoning
  block would make generation much slower and add text to the answers.
- **Greedy decoding** (`do_sample=False`): answers are reproducible; the same
  question gives the same answer on every machine.
- **Automatic device selection**: the model runs on CUDA if available, then
  on Apple MPS, then on CPU. The same code is fast on a Mac or a GPU machine
  and still works anywhere.
- **Errors are caught at the CLI level**: every command validates its
  arguments and turns exceptions into a short `Error: ...` message.

---

## Challenges faced

- **Character positions.** Every chunk must point to the exact range of the
  original file. Reading lines with `splitlines(keepends=True)` or walking the
  text character by character keeps the line breaks in the count; a missing
  `\n` would shift every following position.
- **Never exceeding 2000 characters.** A single oversized chunk invalidates
  the whole output file. Long sections are split by `make_chunks`, and the
  CLI rejects `max_chunk_size` outside `1..2000` (a value of 0 would make the
  splitter loop forever).
- **Vocabulary mismatch.** Questions and code do not use the same words
  (`start the server` vs `run_server`). Identifier splitting, path indexing
  and stemming reduce the gap; true synonyms (`start` vs `run`) remain a
  limit of lexical search.
- **Test files as noise.** They hurt both ranking and generation (see
  *Design decisions*). The problem was found by printing the context actually
  sent to the model, instead of guessing.
- **Missing file types.** The analysis of missed questions showed answers
  located in `CMakeLists.txt`, which was not indexed at first.
- **Small model limits.** Qwen3-0.6B sometimes answers too cautiously, copies
  long code examples, or guesses when the context is wrong. Disabling the
  thinking mode, bounding the context and using a grounding instruction keep
  most answers correct.
- **Generation time.** On CPU, generating 100 answers takes a long time.
  Selecting the MPS device on Apple Silicon brought it to about 6 seconds per
  question.
- **Repository hygiene.** The vLLM corpus contains a string that GitHub's
  secret scanning detected as an API key, which blocked a push. The data is
  excluded with `.gitignore`, as required by the subject.

---

## Example usage

```bash
# Build the index
uv run python -m src index --max_chunk_size 2000
# Ingestion complete! Indices saved under data/processed

# Search for one question
uv run python -m src search "How do I enable LoRA in vLLM?" --k 3
# data/raw/vllm-0.10.1/docs/features/lora.md [0:1884]
# data/raw/vllm-0.10.1/docs/features/lora.md [2072:3835]
# data/raw/vllm-0.10.1/vllm/config/__init__.py [104148:106020]

# Answer one question
uv run python -m src answer "How do I enable LoRA in vLLM?" --k 5
# Model loaded on mps
# Sources:
#   data/raw/vllm-0.10.1/docs/features/lora.md [0:1884]
#   ...
# Answer:
# To enable LoRA in vLLM, you can use the `enable_lora=True` flag when
# initializing the LLM. This flag allows you to include LoRA adapters in
# your model.

# Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# Evaluate it
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# Questions evaluated: 100
# Recall@1: 0.620 (62.0%)
# Recall@3: 0.830 (83.0%)
# Recall@5: 0.850 (85.0%)
# Recall@10: 0.890 (89.0%)

# Generate the answers
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Same pipeline with the Makefile
make run       # index + search + evaluate
make answer    # generate the answers
```

Invalid inputs never crash the program:

```bash
uv run python -m src search "" --k 5          # (no output)
uv run python -m src search "LoRA" --k abc    # Error: k must be an integer
uv run python -m src index --max_chunk_size 0
# Error: max_chunk_size must be an integer between 1 and 2000
uv run python -m src evaluate --student_search_results_path missing.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# Error: [Errno 2] No such file or directory: 'missing.json'
```

---

## Resources

### References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020): <https://arxiv.org/abs/2005.11401>
- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009)
- Manning, Raghavan and Schütze, *Introduction to Information Retrieval*,
  chapters on tokenization, stemming and scoring:
  <https://nlp.stanford.edu/IR-book/>
- `bm25s` library: <https://github.com/xhluca/bm25s>
- Snowball stemmer: <https://snowballstem.org/> and PyStemmer:
  <https://github.com/snowballstem/pystemmer>
- Hugging Face Transformers documentation:
  <https://huggingface.co/docs/transformers>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Python Fire: <https://github.com/google/python-fire>
- Pydantic: <https://docs.pydantic.dev/>
- uv: <https://docs.astral.sh/uv/>
- vLLM documentation (the indexed corpus): <https://docs.vllm.ai/>

### How AI was used

An AI assistant (Claude) was used as a tutor throughout the project:

- **Understanding the concepts**: explanations of RAG, BM25, tokenization,
  stemming, recall@k, chat templates and token budgets, with worked examples.
- **Architecture**: discussing how to split the project into modules and the
  order in which to build them.
- **Code**: the code was written step by step with the assistant's help. Each
  part was explained line by line, tested on small examples and edge cases,
  and modified when needed before moving on.
- **Debugging**: interpreting errors (imports, packaging with uv, git push
  protection, flake8 and mypy messages).
- **Analysis**: writing the script that lists missed questions and
  interpreting the results to choose the improvements, each of which was
  measured with the grader before being kept.
- **Documentation**: drafting this README, then reviewing it.

All design decisions were measured and validated, and every part of the code
can be explained.
