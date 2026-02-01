"""
Simple RAG 命令行入口：编码 PDF、检索上下文、生成回答，可选运行 RAG 评估。
可独立在 rag_systems/simple_rag 下运行，不依赖项目根目录的 helper_functions 或 evaluation。
"""
import os
import sys
import argparse
import time
import json

# 保证从本包目录或任意 cwd 运行时都能正确解析包内导入
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from dotenv import load_dotenv

# 按优先级尝试加载 .env：当前目录 → 本包目录 → 项目根目录（RAG_Techniques）
load_dotenv()
load_dotenv(os.path.join(_PACKAGE_ROOT, ".env"))
_project_root = os.path.normpath(os.path.join(_PACKAGE_ROOT, "..", ".."))
load_dotenv(os.path.join(_project_root, ".env"))
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import config
from lib.helpers import (
    encode_pdf,
    retrieve_context_per_question,
    show_context,
    create_question_answer_from_context_chain,
    answer_question_from_context,
)
from lib.evaluation import evaluate_rag
from langchain_openai import ChatOpenAI


class SimpleRAG:
    """Simple RAG：文档分块、向量检索、基于上下文的问答。"""

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        print("\n--- Initializing Simple RAG Retriever ---")
        start_time = time.time()
        self.vector_store = encode_pdf(
            path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.time_records = {"Chunking": time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")
        self.chunks_query_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": n_retrieved}
        )

    def run(self, query):
        """对给定 query 检索上下文、展示并生成回答。"""
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records["Retrieval"] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")
        show_context(context)
        context_str = "\n\n".join(context)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer_chain = create_question_answer_from_context_chain(llm)
        result = answer_question_from_context(query, context_str, answer_chain)
        print("\n--- 基于检索内容的回答 ---")
        print(result["answer"])


def _validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Encode a PDF and run a simple RAG (optionally evaluate)."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=config.DEFAULT_DOC_PATH,
        help="Path to the PDF file to encode.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=config.DEFAULT_CHUNK_SIZE,
        help="Size of each text chunk.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=config.DEFAULT_CHUNK_OVERLAP,
        help="Overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--n_retrieved",
        type=int,
        default=config.DEFAULT_N_RETRIEVED,
        help="Number of chunks to retrieve per query.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the main cause of climate change?",
        help="Query to run against the retriever.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAG evaluation after the query.",
    )
    parser.add_argument(
        "--evaulate",
        dest="evaluate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return _validate_args(parser.parse_args())


def main(args):
    path = os.path.normpath(args.path.replace("\\", os.sep))
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config.PACKAGE_ROOT, path))

    simple_rag = SimpleRAG(
        path=path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved,
    )
    simple_rag.run(args.query)

    if args.evaluate:
        print("\n--- RAG 评估进行中（约需 1–2 分钟，会调用多次 LLM）---")
        eval_result = evaluate_rag(simple_rag.chunks_query_retriever)
        print("\n--- RAG 评估结果 ---")
        answers_list = eval_result.get("answers", [])
        for i, (q, r) in enumerate(
            zip(eval_result["questions"], eval_result["results"]), 1
        ):
            q_show = (q.strip() or "(无)")[:80]
            print(f"\n问题 {i}: {q_show}")
            if i <= len(answers_list):
                ans_show = (answers_list[i - 1] or "")[:400]
                print(
                    f"Answer: {ans_show}"
                    + ("..." if len(answers_list[i - 1] or "") > 400 else "")
                )
            print(f"评估: {str(r)[:500]}")
        avg = eval_result.get("average_scores")
        if avg:
            print("\n--- 平均分数 (Average Scores) ---")
            print(f"  relevance:   {avg.get('relevance', 0):.2f}")
            print(f"  completeness: {avg.get('completeness', 0):.2f}")
            print(f"  conciseness: {avg.get('conciseness', 0):.2f}")

        os.makedirs(config.DATA_DIR, exist_ok=True)
        out_path = config.EVAL_RESULT_PATH
        to_save = {
            "questions": eval_result["questions"],
            "answers": eval_result.get("answers", []),
            "results": eval_result["results"],
            "average_scores": eval_result.get("average_scores"),
        }
        items = [
            {
                "question": q,
                "answer": eval_result.get("answers", [])[i]
                if i < len(eval_result.get("answers", []))
                else "",
                "evaluation": r,
            }
            for i, (q, r) in enumerate(
                zip(eval_result["questions"], eval_result["results"])
            )
        ]
        to_save["items"] = items
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到: {out_path}")


if __name__ == "__main__":
    main(_parse_args())
