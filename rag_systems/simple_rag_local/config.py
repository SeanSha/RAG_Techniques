"""
Simple RAG Local 配置：完全本地运行，不使用 OpenAI API。
路径与默认参数均相对于本包根目录（rag_systems/simple_rag_local）。
"""
import os

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")
DEFAULT_DOC_PATH = os.path.join(DATA_DIR, "Understanding_Climate_Change.pdf")
EVAL_RESULT_FILENAME = "rag_eval_result.json"
EVAL_RESULT_PATH = os.path.join(DATA_DIR, EVAL_RESULT_FILENAME)

# 本地模型（需先安装 Ollama 并执行 ollama pull <model>）
LOCAL_LLM_MODEL = "llama3.2:3b"
# 本地向量模型（sentence-transformers，首次运行会自动下载）
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_N_RETRIEVED = 2
