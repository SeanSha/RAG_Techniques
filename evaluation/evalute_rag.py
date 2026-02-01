"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
from typing import List, Tuple, Dict, Any

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4-turbo",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4-turbo",
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="gpt-4-turbo",
    include_reason=True
)

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.
    
    Args:
        retriever: The retriever component to evaluate
        num_questions: Number of test questions to generate
    
    Returns:
        Dict containing evaluation metrics
    """
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    
    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?
    
    Provide ratings in JSON format:
    """)
    
    # Create evaluation chain
    eval_chain = (
        eval_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Generate test questions（要求每行一个独立问题，便于按行解析）
    question_gen_prompt = PromptTemplate.from_template(
        "Generate exactly {num_questions} short test questions about climate change. "
        "Output ONLY one question per line, no numbering, no options (e.g. no A) B) C)). "
        "Example format:\n"
        "What is the main cause of climate change?\n"
        "How does deforestation affect the climate?\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    raw_lines = question_chain.invoke({"num_questions": num_questions}).strip().split("\n")
    # 只保留像问题的行：非空、不是选项行（以 - 或 A) B) 等开头）
    questions = [
        line.strip() for line in raw_lines
        if line.strip()
        and not line.strip().startswith(("-", "A)", "B)", "C)", "D)", "a)", "b)", "c)", "d)"))
        and len(line.strip()) > 10
    ][:num_questions]
    if not questions:
        questions = [
            "What is the main cause of climate change?",
            "How do greenhouse gases affect the Earth?",
            "What role do fossil fuels play in climate change?",
        ][:num_questions]
    
    # 用于根据上下文生成回答的 chain
    answer_chain = create_question_answer_from_context_chain(llm)

    # Evaluate each question，并生成每个问题的 RAG 回答
    results = []
    answers = []
    for question in questions:
        # Get retrieval results
        context = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Evaluate results（检索质量打分）
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)

        # 根据检索到的上下文生成回答，供保存到 JSON
        ans_out = answer_question_from_context(question, context_text, answer_chain)
        answers.append(ans_out["answer"])
    
    return {
        "questions": questions,
        "results": results,
        "answers": answers,
        "average_scores": calculate_average_scores(results)
    }

def _parse_eval_json(s: str) -> Dict[str, float]:
    """从评估结果字符串中解析出 relevance/completeness/conciseness（兼容大小写）。"""
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


def calculate_average_scores(results: List[str]) -> Dict[str, float]:
    """Calculate average scores (relevance, completeness, conciseness) across all evaluation results."""
    if not results:
        return {"relevance": 0, "completeness": 0, "conciseness": 0}
    parsed = [_parse_eval_json(r) for r in results]
    n = len(parsed)
    return {
        "relevance": sum(p["relevance"] for p in parsed) / n,
        "completeness": sum(p["completeness"] for p in parsed) / n,
        "conciseness": sum(p["conciseness"] for p in parsed) / n,
    }

if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
