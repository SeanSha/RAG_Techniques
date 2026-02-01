"""
RAG 评估：使用传入的 LLM（如 ChatGroq）对检索结果打分并生成回答。
"""
import json
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .helpers import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
)


def evaluate_rag(retriever, llm, num_questions: int = 5) -> Dict[str, Any]:
    eval_prompt = PromptTemplate.from_template("""
Evaluate the following retrieval results for the question.

Question: {question}
Retrieved Context: {context}

Rate on a scale of 1-5 (5 being best) for:
1. Relevance: How relevant is the retrieved information to the question?
2. Completeness: Does the context contain all necessary information?
3. Conciseness: Is the retrieved context focused and free of irrelevant information?

Reply with ONLY a JSON object (no markdown, no explanation), e.g.:
{{"relevance": 4, "completeness": 3, "conciseness": 4}}
""")
    eval_chain = eval_prompt | llm | StrOutputParser()

    question_gen_prompt = PromptTemplate.from_template(
        "Generate exactly {num_questions} short test questions about climate change. "
        "Output ONLY one question per line, no numbering, no options (e.g. no A) B) C)). "
        "Example format:\n"
        "What is the main cause of climate change?\n"
        "How does deforestation affect the climate?\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    raw_lines = question_chain.invoke({"num_questions": num_questions}).strip().split("\n")
    questions = [
        line.strip()
        for line in raw_lines
        if line.strip()
        and not line.strip().startswith(
            ("-", "A)", "B)", "C)", "D)", "a)", "b)", "c)", "d)")
        )
        and len(line.strip()) > 10
    ][:num_questions]
    if not questions:
        questions = [
            "What is the main cause of climate change?",
            "How do greenhouse gases affect the Earth?",
            "What role do fossil fuels play in climate change?",
        ][:num_questions]

    answer_chain = create_question_answer_from_context_chain(llm)
    results = []
    answers = []
    for question in questions:
        context = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context])
        eval_result = eval_chain.invoke({"question": question, "context": context_text})
        results.append(eval_result)
        ans_out = answer_question_from_context(question, context_text, answer_chain)
        answers.append(ans_out["answer"])

    return {
        "questions": questions,
        "results": results,
        "answers": answers,
        "average_scores": _calculate_average_scores(results),
    }


def _parse_eval_json(s: str) -> Dict[str, float]:
    s = s.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    try:
        d = json.loads(s)
        get_val = lambda key: float(d.get(key, d.get(key.capitalize(), 0)))
        return {
            "relevance": get_val("relevance"),
            "completeness": get_val("completeness"),
            "conciseness": get_val("conciseness"),
        }
    except Exception:
        return {"relevance": 0, "completeness": 0, "conciseness": 0}


def _calculate_average_scores(results: List[str]) -> Dict[str, float]:
    if not results:
        return {"relevance": 0, "completeness": 0, "conciseness": 0}
    parsed = [_parse_eval_json(r) for r in results]
    n = len(parsed)
    return {
        "relevance": sum(p["relevance"] for p in parsed) / n,
        "completeness": sum(p["completeness"] for p in parsed) / n,
        "conciseness": sum(p["conciseness"] for p in parsed) / n,
    }
