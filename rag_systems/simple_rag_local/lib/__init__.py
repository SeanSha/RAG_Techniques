"""Simple RAG Local：完全本地运行（无 OpenAI API）。"""
from .helpers import (
    encode_pdf,
    retrieve_context_per_question,
    show_context,
    create_question_answer_from_context_chain,
    answer_question_from_context,
)
from .evaluation import evaluate_rag

__all__ = [
    "encode_pdf",
    "retrieve_context_per_question",
    "show_context",
    "create_question_answer_from_context_chain",
    "answer_question_from_context",
    "evaluate_rag",
]
