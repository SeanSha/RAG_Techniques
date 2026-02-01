"""Simple RAG Cloud：向量本地，LLM 云端 GPU（Groq 等）。"""
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
