"""
Simple RAG Cloud 命令行入口：LLM 在云端 GPU 上运行（默认 Groq API），向量仍本地。
需设置 GROQ_API_KEY（Groq 有免费额度，GPU 推理在云端）。
"""
import os
import sys
import argparse
import time
import json

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from dotenv import load_dotenv
load_dotenv(_PACKAGE_ROOT)
load_dotenv(os.path.join(_PACKAGE_ROOT, ".env"))
_project_root = os.path.normpath(os.path.join(_PACKAGE_ROOT, "..", ".."))
load_dotenv(os.path.join(_project_root, ".env"))
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

import config
from lib.helpers import (
    encode_pdf,
    retrieve_context_per_question,
    show_context,
    create_question_answer_from_context_chain,
    answer_question_from_context,
)
from lib.evaluation import evaluate_rag
from langchain_groq import ChatGroq


class SimpleRAGCloud:
    """Simple RAG 云端版：向量本地，LLM 云端 GPU（Groq），无需本地显卡。"""

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        print("\n--- Initializing Simple RAG (Cloud GPU) ---")
        start_time = time.time()
        self.vector_store = encode_pdf(
            path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.time_records = {"Chunking": time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")
        self.chunks_query_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": n_retrieved}
        )

    def run(self, query, llm):
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records["Retrieval"] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")
        show_context(context)
        context_str = "\n\n".join(context)
        answer_chain = create_question_answer_from_context_chain(llm)
        print("\n--- 基于检索内容的回答 ---")
        result = answer_question_from_context(query, context_str, answer_chain)
        print()


def _validate_args(args):
    if args.chunk_size <= 0 or args.chunk_overlap < 0 or args.n_retrieved <= 0:
        raise ValueError("chunk_size/n_retrieved 为正整数，chunk_overlap 为非负整数。")
    return args


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Simple RAG 云端版（LLM 用 Groq 云端 GPU）。需设置 GROQ_API_KEY。"
    )
    parser.add_argument("--path", type=str, default=config.DEFAULT_DOC_PATH, help="PDF 路径")
    parser.add_argument("--chunk_size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--n_retrieved", type=int, default=config.DEFAULT_N_RETRIEVED)
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?")
    parser.add_argument("--model", type=str, default=config.CLOUD_LLM_MODEL,
                        help=f"Groq 模型名 (default: {config.CLOUD_LLM_MODEL})")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaulate", dest="evaluate", action="store_true", help=argparse.SUPPRESS)
    return _validate_args(parser.parse_args())


def main(args):
    path = os.path.normpath(args.path.replace("\\", os.sep))
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config.PACKAGE_ROOT, path))

    if not os.environ.get("GROQ_API_KEY"):
        print("未设置 GROQ_API_KEY。请在 .env 中设置，或：export GROQ_API_KEY=你的key")
        raise SystemExit(1)

    llm = ChatGroq(model=args.model, temperature=0)

    simple_rag = SimpleRAGCloud(
        path=path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved,
    )
    simple_rag.run(args.query, llm=llm)

    if args.evaluate:
        print("\n--- RAG 评估进行中（云端 LLM，会调用多次）---")
        eval_result = evaluate_rag(simple_rag.chunks_query_retriever, llm=llm)
        print("\n--- RAG 评估结果 ---")
        answers_list = eval_result.get("answers", [])
        for i, (q, r) in enumerate(zip(eval_result["questions"], eval_result["results"]), 1):
            print(f"\n问题 {i}: {(q.strip() or '(无)')[:80]}")
            if i <= len(answers_list):
                ans = (answers_list[i - 1] or "")[:400]
                print(f"Answer: {ans}" + ("..." if len(answers_list[i - 1] or "") > 400 else ""))
            print(f"评估: {str(r)[:500]}")
        avg = eval_result.get("average_scores")
        if avg:
            print("\n--- 平均分数 (Average Scores) ---")
            print(f"  relevance:   {avg.get('relevance', 0):.2f}")
            print(f"  completeness: {avg.get('completeness', 0):.2f}")
            print(f"  conciseness: {avg.get('conciseness', 0):.2f}")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        to_save = {
            "questions": eval_result["questions"],
            "answers": eval_result.get("answers", []),
            "results": eval_result["results"],
            "average_scores": eval_result.get("average_scores"),
            "items": [
                {"question": q, "answer": eval_result.get("answers", [])[i] if i < len(eval_result.get("answers", [])) else "", "evaluation": r}
                for i, (q, r) in enumerate(zip(eval_result["questions"], eval_result["results"]))
            ],
        }
        with open(config.EVAL_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到: {config.EVAL_RESULT_PATH}")


if __name__ == "__main__":
    main(_parse_args())
