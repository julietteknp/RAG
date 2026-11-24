"""Pydantic data models exchanged between the RAG pipeline stages."""

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A location in the corpus: a file and a character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without its expected sources or answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its expected sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question, plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output file of the search_dataset command."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output file of the answer_dataset command."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """A piece of a corpus file, ready to be indexed."""

    search_text: str
