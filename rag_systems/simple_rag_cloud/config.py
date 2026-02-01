"""
Simple RAG Cloud 配置：LLM 在云端 GPU 上运行（默认 Groq API）。
路径与默认参数均相对于本包根目录（rag_systems/simple_rag_cloud）。
"""
import os

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")
DEFAULT_DOC_PATH = os.path.join(DATA_DIR, "Understanding_Climate_Change.pdf")
EVAL_RESULT_FILENAME = "rag_eval_result.json"
EVAL_RESULT_PATH = os.path.join(DATA_DIR, EVAL_RESULT_FILENAME)

# 云端 LLM（Groq 在云端 GPU 上推理，需 GROQ_API_KEY，有免费额度）
CLOUD_LLM_MODEL = "llama-3.1-8b-instant"
# 向量仍用本地 sentence-transformers，不占云端费用
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_N_RETRIEVED = 2
