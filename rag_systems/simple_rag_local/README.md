# Simple RAG Local（完全本地版）

在 `simple_rag` 基础上复制的**完全本地**版本：向量与生成均在本地完成，**不使用 OpenAI API**，无 API 费用。

---

## 与 simple_rag 的区别

| 项目 | simple_rag | simple_rag_local |
|------|------------|------------------|
| 向量（建索引） | OpenAI Embeddings（云端） | **sentence-transformers**（本地） |
| 生成回答 / 评估 | OpenAI gpt-4o-mini（云端） | **Ollama** 本地模型 |
| 环境变量 | 需要 `OPENAI_API_KEY` | **不需要** API Key |
| 费用 | 按 token 计费 | **无 API 费用** |

---

## 运行流程与组件（初学者向）

执行 `python main.py --query "..."` 时，程序按以下顺序运行，用到的组件如下。

### 1. 入口与参数（main.py）

- 解析命令行：`--path`、`--query`、`--model`、`--evaluate`、`--ollama-host` 等。
- **WSL 下**：若未设置 `OLLAMA_HOST`，从 `/etc/resolv.conf` 读取 Windows 主机 IP 并写入环境变量，供后续连 Ollama 使用。
- 创建 **ChatOllama**（`langchain_ollama`）：本地对话模型，默认 `llama3.2:3b`。

### 2. 初始化 RAG：建向量索引（main.py → lib/helpers.py）

- **SimpleRAGLocal(path)** 被调用。
- **encode_pdf(path)**（`lib/helpers.py`）：
  - **PyPDFLoader**：按页加载 PDF。
  - **RecursiveCharacterTextSplitter**：按 `chunk_size` / `chunk_overlap` 分块。
  - **HuggingFaceEmbeddings**：用 `config.EMBEDDING_MODEL`（默认 `all-MiniLM-L6-v2`）对每块做向量，本地运行，无 API。
  - **FAISS.from_documents**：把向量存入 FAISS 索引。
- **vector_store.as_retriever(k=n_retrieved)**：得到检索器，之后用「问题」查「最相关的 n 块」。
- 终端输出：`Chunking Time: x.xx seconds`。

### 3. 一次查询：检索 → 展示 → 生成回答（main.py → lib/helpers.py）

- **simple_rag.run(query, llm)**：
  - **retrieve_context_per_question(query, retriever)**：用问题向量在 FAISS 里检索，得到若干块文本（字符串列表）。
  - 输出：`Retrieval Time: x.xx seconds`，以及 **Context 1 / Context 2** 的原文。
  - **create_question_answer_from_context_chain(llm)**：构造 LangChain 链（Prompt + LLM + StrOutputParser），要求「仅根据给定 context 回答 question」。
  - **answer_question_from_context(query, context_str, chain)**：把「问题 + 检索到的上下文」喂给链；用 **stream()** 流式调用 Ollama，边生成边打印；得到最终回答。
- 终端输出：`--- 基于检索内容的回答 ---` 以及模型生成的回答。

### 4. 可选：RAG 评估（main.py → lib/evaluation.py）

- 仅当传入 **--evaluate** 时执行。
- **evaluate_rag(retriever, llm)**（`lib/evaluation.py`）：
  - 用 **question_chain**（同一 Ollama）生成若干条「气候相关」测试问题（默认 5 条）。
  - 对**每个问题**：用 retriever 检索上下文 → 用 **eval_chain** 对「问题 + 上下文」打 relevance / completeness / conciseness（1–5 分，JSON）→ 用 **answer_chain** 根据上下文生成回答。
  - 汇总各题打分，计算平均；把问题、回答、打分、平均分写入 **data/rag_eval_result.json**。
- 终端输出：每个问题的回答与评估 JSON，以及平均分数。

### 组件与文件对应关系

| 步骤 | 所用组件 | 所在文件 |
|------|----------|----------|
| 配置与路径 | PACKAGE_ROOT, DATA_DIR, DEFAULT_DOC_PATH, LOCAL_LLM_MODEL, EMBEDDING_MODEL | config.py |
| PDF → 分块 | PyPDFLoader, RecursiveCharacterTextSplitter | lib/helpers.py |
| 分块 → 向量 | HuggingFaceEmbeddings（sentence-transformers） | lib/helpers.py |
| 向量存储与检索 | FAISS, as_retriever() | lib/helpers.py |
| 问答链（根据上下文回答） | PromptTemplate + LLM + StrOutputParser，流式输出 | lib/helpers.py |
| 本地 LLM | ChatOllama（Ollama） | main.py，传入 helpers / evaluation |
| 评估（问题生成、打分、平均） | evaluate_rag, question_chain, eval_chain, answer_chain | lib/evaluation.py |

---

## 前置条件

1. **安装 Ollama**（https://ollama.com），并拉取模型，例如：  
   `ollama pull llama3.2:3b`  
   默认模型可在 `config.py` 的 `LOCAL_LLM_MODEL` 或命令行 `--model` 中修改。
2. **Python 依赖**：`pip install -r requirements.txt`  
   首次运行会下载 sentence-transformers 模型（约几百 MB），之后本地缓存。

---

## 目录结构

```
rag_systems/simple_rag_local/
├── config.py          # 路径、默认 PDF、本地模型名、向量模型名
├── main.py            # 命令行入口，组装 SimpleRAGLocal + 评估
├── lib/
│   ├── helpers.py     # 编码 PDF、检索、展示上下文、问答链、流式回答
│   └── evaluation.py  # 评估：生成问题、打分、生成回答、写 JSON
├── data/              # 待索引 PDF、rag_eval_result.json
├── requirements.txt
└── README.md
```

---

## 运行方式

```bash
cd rag_systems/simple_rag_local
# 将 PDF 放入 data/ 或通过 --path 指定
python main.py --query "what is the main cause of climate change?"
```

- 指定模型：`python main.py --model qwen2.5:3b --query "你的问题"`
- 带评估：`python main.py --query "..." --evaluate`

---

## 默认配置（config.py）

- **LOCAL_LLM_MODEL**：`llama3.2:3b`
- **EMBEDDING_MODEL**：`sentence-transformers/all-MiniLM-L6-v2`

---

## WSL 下使用 Windows 上的 Ollama

若 **Ollama 在 Windows**、**Python 在 WSL**：

1. **Windows**：设置用户环境变量 `OLLAMA_HOST=0.0.0.0`，并**重启 Ollama**，使 WSL 能通过 IP 访问。
2. **WSL**：程序会尝试从 `/etc/resolv.conf` 取 Windows 主机 IP；若仍 Connection refused，可手动指定：  
   `export OLLAMA_HOST=http://<Windows的IP>:11434`  
   或：`python main.py --ollama-host http://<IP>:11434 --query "..."`  
   Windows 的 IP 可在 PowerShell 中 `ipconfig` 查看「vEthernet (WSL)」的 IPv4。
