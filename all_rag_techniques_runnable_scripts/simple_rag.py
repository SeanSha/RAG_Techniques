import os
import sys
import argparse
import time
from dotenv import load_dotenv

# 项目根目录（脚本在 all_rag_techniques_runnable_scripts 时，父目录即根）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _project_root)

# 先加载当前目录，再加载项目根目录的 .env，确保子目录运行时也能读到 OPENAI_API_KEY
load_dotenv()
load_dotenv(os.path.join(_project_root, '.env'))
if os.getenv('OPENAI_API_KEY'):
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from helper_functions import (
    encode_pdf,
    retrieve_context_per_question,
    show_context,
    create_question_answer_from_context_chain,
    answer_question_from_context,
)
from langchain_openai import ChatOpenAI
# evaluation 含 deepeval，仅在 --evaluate 时导入


class SimpleRAG:
    """
    A class to handle the Simple RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        Initializes the SimpleRAGRetriever by encoding the PDF document and creating the retriever.

        Args:
            path (str): Path to the PDF file to encode.
            chunk_size (int): Size of each text chunk (default: 1000).
            chunk_overlap (int): Overlap between consecutive chunks (default: 200).
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
        """
        print("\n--- Initializing Simple RAG Retriever ---")

        # Encode the PDF document into a vector store using OpenAI embeddings
        start_time = time.time()
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Create a retriever from the vector store
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for retrieval
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(context)

        # 根据检索到的上下文用 LLM 生成一段回答并打印
        context_str = "\n\n".join(context)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer_chain = create_question_answer_from_context_chain(llm)
        result = answer_question_from_context(query, context_str, answer_chain)
        print("\n--- 基于检索内容的回答 ---")
        print(result["answer"])


# Function to validate command line inputs
def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks (default: 200).")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to evaluate the retriever's performance (default: False).")

    # Parse and validate arguments
    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the SimpleRAGRetriever class
def main(args):
    # 规范化路径：Windows 反斜杠在 WSL/Linux 下会出错，统一为当前系统的分隔符并解析为绝对路径
    path = os.path.normpath(args.path.replace("\\", os.sep))
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Initialize the SimpleRAGRetriever
    simple_rag = SimpleRAG(
        path=path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Retrieve context based on the query
    simple_rag.run(args.query)

    # Evaluate the retriever's performance on the query (if requested)
    if args.evaluate:
        import json
        from evaluation.evalute_rag import evaluate_rag
        print("\n--- RAG 评估进行中（约需 1–2 分钟，会调用多次 LLM）---")
        eval_result = evaluate_rag(simple_rag.chunks_query_retriever)
        print("\n--- RAG 评估结果 ---")
        answers_list = eval_result.get("answers", [])
        for i, (q, r) in enumerate(zip(eval_result["questions"], eval_result["results"]), 1):
            q_show = (q.strip() or "(无)")[:80]
            print(f"\n问题 {i}: {q_show}")
            if i <= len(answers_list):
                ans_show = (answers_list[i - 1] or "")[:400]
                print(f"Answer: {ans_show}" + ("..." if len(answers_list[i - 1] or "") > 400 else ""))
            print(f"评估: {str(r)[:500]}")
        avg = eval_result.get("average_scores")
        if avg:
            print("\n--- 平均分数 (Average Scores) ---")
            print(f"  relevance:   {avg.get('relevance', 0):.2f}")
            print(f"  completeness: {avg.get('completeness', 0):.2f}")
            print(f"  conciseness: {avg.get('conciseness', 0):.2f}")
        # 保存到项目 data 目录（含问题、回答、评估），方便后续查看
        out_path = os.path.join(_project_root, "data", "rag_eval_result.json")
        to_save = {
            "questions": eval_result["questions"],
            "answers": eval_result.get("answers", []),
            "results": eval_result["results"],
            "average_scores": eval_result.get("average_scores"),
        }
        # 同时保存为「一问一答一评估」的列表，类似 q_a.json
        items = [
            {
                "question": q,
                "answer": eval_result.get("answers", [])[i] if i < len(eval_result.get("answers", [])) else "",
                "evaluation": r,
            }
            for i, (q, r) in enumerate(zip(eval_result["questions"], eval_result["results"]))
        ]
        to_save["items"] = items
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到: {out_path}")


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
