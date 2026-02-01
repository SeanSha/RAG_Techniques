# 云端 GPU 租用与部署指南（最小验证）

目标：**在云端租一块 GPU，部署 Ollama + 开源模型，用 simple_rag_local 做一次最简单的 RAG 验证**。  
这样你可以自由换模型、后续做微调，而不依赖 OpenAI/Groq 等 API。

---

## 第一次操作清单（按顺序做）

下面是一套「从零到跑通」的步骤，按顺序执行即可。

### 阶段 1：在 RunPod 开一台 GPU 机子

| 步骤 | 操作 |
|------|------|
| 1 | 打开 https://www.runpod.io ，注册并登录。 |
| 2 | **Billing** → 充值（如 10 美元），用于按小时扣费。 |
| 3 | **Pods** → **+ Deploy**。 |
| 4 | **Select GPU**：选 **RTX 3060** 或 **T4**（便宜、够用）。 |
| 5 | **Container Image**：选 **RunPod Pytorch 2.1** 或 **Ubuntu 22.04**。 |
| 6 | **Disk**：改成 **50 GB** 或以上。 |
| 7 | 点 **Deploy**，等状态变成 **Running**。 |
| 8 | 点进该 Pod，记下 **SSH** 里的命令（或直接用下面的 **Connect → Start Web Terminal**）。 |

### 阶段 2：进实例（二选一）

- **方式 A（推荐）**：在 RunPod 该 Pod 页面点 **Connect** → **Start Web Terminal**，浏览器里会打开一个终端，相当于已经 SSH 进实例。
- **方式 B**：在本机用 RunPod 给的 SSH 命令，例如：  
  `ssh root@xxx-xxx.runpod.io -p 12345 -i ~/.ssh/your_key`

### 阶段 3：在实例里装 Ollama 和模型

在**云端实例的终端**里逐条执行（复制粘贴即可）：

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama 并拉取小模型（约 2GB，几分钟）
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 10
ollama pull llama3.2:3b

# 确认模型已在
ollama list
```

看到 `llama3.2:3b` 即表示成功。

### 阶段 4：在实例里装 Python 和项目依赖

继续在**同一终端**执行：

```bash
# 装 Miniconda（若没有 conda）
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# 创建环境并安装 simple_rag_local 的依赖（先建目录，依赖稍后装）
mkdir -p ~/RAG_Techniques/rag_systems/simple_rag_local
```

接下来要把 **simple_rag_local** 的代码和 **requirements.txt** 弄到实例上（见阶段 5），然后再回来执行：

```bash
cd ~/RAG_Techniques/rag_systems/simple_rag_local
pip install -r requirements.txt
```

### 阶段 5：把 simple_rag_local 代码和 PDF 弄到实例上（二选一）

**方式 A：本机用 SCP 上传（你在本机是 Windows 可用 WSL）**

在本机**新开一个终端**，进入项目目录后执行（把 `<实例IP>`、`<端口>`、`<私钥路径>` 换成 RunPod 给你的）：

```bash
cd /mnt/d/Coding/RAGExamples/RAG_Techniques   # 或你项目所在路径

# 上传整个 rag_systems（含 simple_rag_local）
scp -P <端口> -r -i <私钥路径> rag_systems root@<实例IP>:~/RAG_Techniques/

# 上传 PDF 到 simple_rag_local 的 data
scp -P <端口> -i <私钥路径> rag_systems/simple_rag_local/data/Understanding_Climate_Change.pdf root@<实例IP>:~/RAG_Techniques/rag_systems/simple_rag_local/data/
```

若 RunPod 用的是 **Web Terminal** 且没有给你 SSH 密钥，用下面的方式 B。

**方式 B：在实例上直接创建最小文件（不依赖本机上传）**

在**云实例终端**里执行（复制整段）：

```bash
mkdir -p ~/RAG_Techniques/rag_systems/simple_rag_local/data
mkdir -p ~/RAG_Techniques/rag_systems/simple_rag_local/lib
```

然后把本仓库里 **simple_rag_local** 下的这些文件内容，在实例上逐个创建出来（用 `nano` 或 `cat > 文件名 << 'EOF'` 粘贴）：

- `config.py`
- `main.py`
- `lib/helpers.py`
- `lib/evaluation.py`
- `lib/__init__.py`
- `requirements.txt`

PDF 若本机能上传就上传；若暂时没有，可先**跳过 PDF**，用 `--path` 指向实例上任意一个小 PDF 做「能跑通」的验证。

更省事的做法：若项目已在 **GitHub**，在实例上直接：

```bash
cd ~
git clone https://github.com/<你的用户名>/<仓库名>.git RAG_Techniques
cd RAG_Techniques/rag_systems/simple_rag_local
# 再把 PDF 放到 data/ 或之后用 --path 指定
```

### 阶段 6：跑最小验证

在**云实例终端**里（确保在 simple_rag_local 目录且已 `pip install -r requirements.txt`）：

```bash
cd ~/RAG_Techniques/rag_systems/simple_rag_local
conda activate rag   # 若前面用 miniconda 装了 rag 环境
# 若没建 conda 环境，可直接：pip install -r requirements.txt 后执行下一行
python main.py --query "what is the main cause of climate change?"
```

若看到：**Chunking Time** → **Retrieval Time** → **Context 1 / Context 2** → **基于检索内容的回答** 和一段英文回答，说明**云端 GPU + Ollama + simple_rag_local 的最小验证已跑通**。

（若没有 PDF，会报错找不到文件，需要先完成阶段 5 的 PDF 上传或 `--path` 指定。）

### 阶段 7：用完关机省钱

RunPod 里对该 Pod 点 **Stop** 或 **Terminate**，停止计费。下次再用可重新 **Deploy** 或开机，代码若没删就还在；若删了再按阶段 5 传一次即可。

---

以下为分步说明与可选内容（安装脚本、其他平台等）。

---

## 一、选平台并开一台 GPU 实例（以 RunPod 为例）

### 1. 注册与计费

1. 打开 https://www.runpod.io ，注册账号。
2. 充值少量金额（如 10 美元）用于按小时计费。
3. 进入 **Pods** → **Deploy**。

### 2. 创建 GPU 实例

- **GPU 类型**：做最小验证选 **1× RTX 3060（12GB）** 或 **1× T4（16GB）** 即可，时租约 0.2–0.5 美元/小时。
- **镜像**：选 **RunPod PyTorch** 或 **Ubuntu 22.04**（官方模板即可）。
- **区域**：选延迟低的（如 US / EU）。
- **磁盘**：建议 50GB+，用于装 Ollama 和模型。
- 创建后记下 **SSH 命令**（如 `ssh root@xxx.runpod.io -p xxxxx -i your_key`）和 **Web 终端**入口（可选）。

---

## 二、登录实例并安装环境

### 1. SSH 登录

```bash
ssh root@<你的实例IP> -p <端口> -i <你的私钥路径>
```

或使用 RunPod 网页上的 **Connect** → **Start Web Terminal**。

### 2. 安装 Ollama（Linux）

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

安装完成后启动服务（多数镜像会随系统启动）：

```bash
ollama serve &
ollama pull llama3.2:3b
```

验证：`ollama list` 应能看到 `llama3.2:3b`。

### 3. 安装 Python 与项目依赖

```bash
# 若没有 conda，可先装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# 创建环境并安装依赖
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt
```

`requirements.txt` 使用 **simple_rag_local** 目录下的（见下一节「上传代码」）。

---

## 三、上传代码与数据（simple_rag_local）

在云端跑的就是 **simple_rag_local** 这一套代码（Ollama + sentence-transformers），不是本目录的 Groq 版。

### 方式 A：本机用 SCP/rsync 上传

在本机（Windows 可用 WSL 或 Git Bash）执行：

```bash
# 上传整个项目（或只传 rag_systems/simple_rag_local）
scp -P <端口> -r -i <私钥> /path/to/RAG_Techniques root@<实例IP>:~/
```

上传后 SSH 进实例：

```bash
cd ~/RAG_Techniques/rag_systems/simple_rag_local
```

### 方式 B：在实例上 git clone（若项目在 GitHub）

```bash
cd ~
git clone <你的仓库URL> RAG_Techniques
cd RAG_Techniques/rag_systems/simple_rag_local
```

### 准备 PDF

把用于 RAG 的 PDF 放到 `data/` 下，或通过 `--path` 指定路径。例如从本机上传：

```bash
scp -P <端口> -i <私钥> /path/to/Understanding_Climate_Change.pdf root@<实例IP>:~/RAG_Techniques/rag_systems/simple_rag_local/data/
```

---

## 四、在云端跑一次最小验证

在实例上（已 `conda activate rag` 且 `cd` 到 `simple_rag_local`）：

```bash
# Ollama 在本机，无需 OLLAMA_HOST
python main.py --query "what is the main cause of climate change?"
```

若看到：Chunking → Retrieval → Context 1/2 → 基于检索内容的回答，说明 **云端 GPU + Ollama + simple_rag_local** 的最小验证已跑通。

可选再跑评估：

```bash
python main.py --query "what is the main cause of climate change?" --evaluate
```

---

## 五、可选：一键安装脚本（Ubuntu 22.04）

在实例上新建 `setup_cloud_gpu.sh`，内容如下（按需修改），然后 `chmod +x setup_cloud_gpu.sh && ./setup_cloud_gpu.sh`：

```bash
#!/bin/bash
set -e
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
ollama pull llama3.2:3b

# 若用 conda
if ! command -v conda &>/dev/null; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p $HOME/miniconda3
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
conda create -n rag python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag
cd /root  # 或你 clone/上传的目录
pip install -r RAG_Techniques/rag_systems/simple_rag_local/requirements.txt
echo "Done. Run: conda activate rag && cd RAG_Techniques/rag_systems/simple_rag_local && python main.py --query '...'"
```

---

## 六、其他平台（简要）

| 平台 | 创建实例后 | 备注 |
|------|------------|------|
| **Lambda Labs** | SSH 进实例，同上安装 Ollama + conda + simple_rag_local | 按小时/包月，文档清晰 |
| **Vast.ai** | 选带 GPU 的镜像，SSH 进容器，同上装 Ollama 与依赖 | 价格低，稳定性一般 |
| **Google Colab Pro** | 新建 Notebook，运行时选 GPU，`!curl -fsSL https://ollama.com/install.sh \| sh` 等需在 Colab 里适配 | 适合轻度、不想配 SSH |

逻辑相同：**GPU 实例 + Linux + Ollama + simple_rag_local 代码 + PDF**，跑 `main.py --query "..."` 做最小验证。

---

## 七、小结

- **在云端租 GPU**：选 RunPod / Lambda / Vast.ai 等，开一台 GPU 实例（Ubuntu 22.04 即可）。
- **部署**：实例上安装 Ollama → `ollama pull llama3.2:3b` → 安装 Python 依赖（simple_rag_local 的 requirements.txt）。
- **验证**：上传或 clone **simple_rag_local** 与 PDF，在实例上执行 `python main.py --query "what is the main cause of climate change?"`。
- **后续**：可换 `ollama pull` 其他模型、改 `config.LOCAL_LLM_MODEL` 或 `--model`，或在该实例上做微调后再用 Ollama 加载。

本目录下的 **Groq 版（main.py）** 是「不想管 GPU 机器、只调 API」时的备选；**你要的自由选模型/微调**，用 **云端 GPU + simple_rag_local** 按本指南部署即可。
