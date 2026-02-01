# Simple RAG Cloud（云端相关用法）

本目录提供两种「云端」用法，按你的目标二选一：

| 目标 | 做法 | 看哪里 |
|------|------|--------|
| **不想管机器，只想调 API 用云端 LLM** | 用本目录的 **Groq API**（现有 `main.py`） | 下面「Groq API」一节 |
| **在云端租一块 GPU，自己部署 Ollama/开源模型、做微调、做最小验证** | 在云 GPU 虚拟机上跑 **simple_rag_local** | **[CLOUD_GPU_DEPLOY.md](./CLOUD_GPU_DEPLOY.md)**（部署与最小验证步骤） |

---

## Groq API（调 API 用云端 LLM，免运维）

- **含义**：你本机跑脚本，**生成回答**时调用 **Groq** 的 API，模型在 Groq 的 GPU 上跑。
- **优点**：免运维、有免费额度、响应快；只需注册拿 API Key，不用自己开虚拟机。
- **步骤**：
  1. 打开 https://console.groq.com ，注册并进入 API Keys。
  2. 新建一个 Key，复制。
  3. 在项目根目录或本目录的 `.env` 里写：`GROQ_API_KEY=你的key`。
  4. 安装依赖：`pip install -r requirements.txt`
  5. 运行：`python main.py --query "what is the main cause of climate change?"`
- **默认模型**：`llama-3.1-8b-instant`（可在 `config.py` 或 `--model` 修改）。

---

## 云端 GPU 方案推荐（开通资源时参考）

需要「GPU 在云端算」时，可以选下面两类之一。

### 一、云 LLM API（GPU 在服务端，免运维）

| 服务 | 说明 | 适合 |
|------|------|------|
| **Groq** | 本仓库默认；免费额度 + 按量计费，延迟低 | 想快速试、不想管服务器 |
| **Together** | 多种开源模型，按 token 计费 | 想换不同模型、做实验 |
| **OpenAI** | gpt-4o-mini 等，按 token 计费 | 要最好效果、可接受费用 |
| **其他** | 如 Replicate、Fireworks 等 | 按需选用 |

**本目录 (simple_rag_cloud)** 已接好 **Groq**；若改用 Together/OpenAI，只需在 `main.py` 里把 `ChatGroq` 换成对应 LangChain 的 Chat 类并改环境变量即可。

### 二、云 GPU 虚拟机（自己跑 Ollama/模型）

适合：想完全自管模型、长时间占满 GPU、或 API 不满足需求时。

| 服务 | 说明 | 适合 |
|------|------|------|
| **RunPod** | 按小时租 GPU 实例，可装 Ollama | 长时间跑、要完全自控 |
| **Lambda Labs** | GPU 云主机，按小时/包月 | 稳定做实验、小团队 |
| **Vast.ai** | 散户 GPU 租用，价格低 | 对价格敏感、能接受不稳定 |
| **Google Colab Pro** | 带 GPU 的 Notebook，月费 | 轻度使用、不想配环境 |
| **AWS / GCP / Azure** | 按需开 GPU 实例 | 已有云账号、要和企业环境一致 |

若选「云 GPU 虚拟机」：**按 [CLOUD_GPU_DEPLOY.md](./CLOUD_GPU_DEPLOY.md) 一步步操作**——在虚拟机上安装 Ollama、装依赖、跑 **simple_rag_local** 做最小验证；后续可自由换模型或做微调。

---

## 与 simple_rag_local 的区别

| 项目 | simple_rag_local | simple_rag_cloud |
|------|------------------|------------------|
| 向量（建索引） | sentence-transformers（本地） | sentence-transformers（本地） |
| 生成回答 / 评估 | Ollama（本机或本机可访问的机器） | **Groq API**（云端 GPU） |
| 本机需求 | 能跑 Ollama 或能连到 Ollama | 无显卡要求，能联网即可 |
| 费用 | 无 API 费用 | Groq 有免费额度，超出按量 |

---

## 目录结构

```
rag_systems/simple_rag_cloud/
├── config.py       # CLOUD_LLM_MODEL（Groq 模型名）、路径、向量模型名
├── main.py         # 入口，使用 ChatGroq
├── lib/
│   ├── helpers.py  # 向量 + 问答链（与 local 相同，LLM 由 main 传入）
│   └── evaluation.py
├── data/
├── requirements.txt
└── README.md
```

---

## 运行方式

```bash
cd rag_systems/simple_rag_cloud
# 确保 .env 中有 GROQ_API_KEY=你的key
pip install -r requirements.txt
python main.py --query "what is the main cause of climate change?"
```

- 指定模型：`python main.py --model llama-3.1-70b-versatile --query "你的问题"`
- 带评估：`python main.py --query "..." --evaluate`

---

## 默认配置（config.py）

- **CLOUD_LLM_MODEL**：`llama-3.1-8b-instant`（Groq 模型名）
- **EMBEDDING_MODEL**：`sentence-transformers/all-MiniLM-L6-v2`（本地，与 local 一致）

总结：
- **本目录 main.py**：向量本地 + LLM 用 **Groq API**（适合不想管 GPU 机器时）。
- **云端租 GPU、自己部署模型/微调**：看 **[CLOUD_GPU_DEPLOY.md](./CLOUD_GPU_DEPLOY.md)**，在云虚拟机上跑 **simple_rag_local** 做最小验证。
