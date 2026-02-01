"""
Simple RAG 子系统配置：路径与默认参数。
所有路径均相对于本包根目录（rag_systems/simple_rag），不依赖项目根目录。
"""
import os

# 本包根目录（即 rag_systems/simple_rag）
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录：存放待索引 PDF 与评估结果
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")

# 默认文档路径（用户可将 PDF 放入 data/ 或通过 --path 指定）
DEFAULT_DOC_PATH = os.path.join(DATA_DIR, "Understanding_Climate_Change.pdf")

# 评估结果输出文件
EVAL_RESULT_FILENAME = "rag_eval_result.json"
EVAL_RESULT_PATH = os.path.join(DATA_DIR, EVAL_RESULT_FILENAME)

# 默认分块与检索参数
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_N_RETRIEVED = 2
