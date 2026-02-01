"""
Simple RAG Local 命令行入口：完全本地运行，不使用 OpenAI API。
- 向量：sentence-transformers（HuggingFaceEmbeddings）
- 生成：Ollama 本地模型（ChatOllama）
需先安装 Ollama 并执行 ollama pull <model>。
"""
import os
import sys
import argparse
import time
import json

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


def _ensure_ollama_host_wsl():
    """WSL 下若未设置 OLLAMA_HOST，用 /etc/resolv.conf 的 nameserver 作为 Windows 主机地址。必须在 import ChatOllama 之前调用。"""
    if os.environ.get("OLLAMA_HOST", "").strip():
        return
    if sys.platform != "linux":
        return
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    ip = line.split()[1].strip()
                    os.environ["OLLAMA_HOST"] = f"http://{ip}:11434"
                    return
    except Exception:
        pass


# 在 import ChatOllama 之前设置 OLLAMA_HOST，否则 ollama 客户端可能仍连 localhost
_ensure_ollama_host_wsl()

import config
from lib.helpers import (
    encode_pdf,
    retrieve_context_per_question,
    show_context,
    create_question_answer_from_context_chain,
    answer_question_from_context,
)
from lib.evaluation import evaluate_rag
from langchain_ollama import ChatOllama
import httpx


class SimpleRAGLocal:
    """Simple RAG 完全本地版：本地向量 + 本地 LLM，无 API 费用。"""

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        print("\n--- Initializing Simple RAG (Local) ---")
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
        """对给定 query 检索上下文、展示并用本地 LLM 生成回答。"""
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records["Retrieval"] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")
        show_context(context)
        context_str = "\n\n".join(context)
        answer_chain = create_question_answer_from_context_chain(llm)
        print("\n--- 基于检索内容的回答 ---")
        result = answer_question_from_context(query, context_str, answer_chain)
        print()  # 流式输出后补换行


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
        description="Simple RAG 完全本地版（无 OpenAI API）。需先安装 Ollama 并 pull 模型。"
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
        "--model",
        type=str,
        default=config.LOCAL_LLM_MODEL,
        help=f"Ollama model name (default: {config.LOCAL_LLM_MODEL}).",
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
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.environ.get("OLLAMA_HOST", ""),
        help="Ollama 服务地址，如 http://172.x.x.x:11434（WSL 连 Windows 时填 Windows 的 IP）。",
    )
    return _validate_args(parser.parse_args())


def main(args):
    path = os.path.normpath(args.path.replace("\\", os.sep))
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config.PACKAGE_ROOT, path))

    # Ollama 地址：命令行 --ollama-host 优先，否则用环境变量 OLLAMA_HOST（WSL 下脚本已尝试自动设置）
    if getattr(args, "ollama_host", ""):
        os.environ["OLLAMA_HOST"] = args.ollama_host.strip()
    llm = ChatOllama(model=args.model, temperature=0)

    simple_rag = SimpleRAGLocal(
        path=path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved,
    )
    try:
        simple_rag.run(args.query, llm=llm)
    except httpx.ConnectError as e:
        print("\n[Ollama 连接失败] Connection refused。若你在 WSL 且 Ollama 在 Windows 上：")
        print("  1. 在 Windows 上确认 Ollama 已启动（托盘或开始菜单）。")
        print("  2. 在 WSL 执行: grep nameserver /etc/resolv.conf  # 记下 IP")
        print("  3. 运行: export OLLAMA_HOST=http://<上一步的IP>:11434")
        print("  4. 或: python main.py --ollama-host http://<IP>:11434 --query \"...\"")
        print("  5. 若仍失败，在 Windows 中设置环境变量 OLLAMA_HOST=0.0.0.0 后重启 Ollama，使 WSL 可访问。")
        raise SystemExit(1) from e

    if args.evaluate:
        print("\n--- RAG 评估进行中（本地 LLM，会调用多次）---")
        eval_result = evaluate_rag(simple_rag.chunks_query_retriever, llm=llm)
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
