# Simple RAG（独立副本）

本目录是 `all_rag_techniques_runnable_scripts/simple_rag.py` 的**独立副本**，按软件工程习惯组织：配置与数据路径集中、代码与数据分离、可单独运行且不修改原项目行为。

## 目录结构

```
rag_systems/simple_rag/
├── config.py          # 路径与默认参数（PACKAGE_ROOT、DATA_DIR、默认文档路径等）
├── main.py            # 命令行入口
├── lib/
│   ├── __init__.py
│   ├── helpers.py     # 编码 PDF、检索、展示上下文、问答链
│   └── evaluation.py  # RAG 评估（问题生成、打分、平均）
├── data/              # 数据目录：放置 PDF，评估结果写于此
│   └── .gitkeep
├── requirements.txt  # 本子系统依赖
└── README.md
```

## 运行方式

1. **安装依赖**（可选，若已在项目根安装则可复用）  
   ```bash
   pip install -r requirements.txt
   ```

2. **环境变量**  
   在 `rag_systems/simple_rag` 或项目根目录放置 `.env`，设置 `OPENAI_API_KEY`。

3. **准备文档**  
   将待索引的 PDF 放入 `data/`，或通过 `--path` 指定路径。默认期望文件为 `data/Understanding_Climate_Change.pdf`。

4. **运行**  
   - 在 **本目录** 下执行：  
     ```bash
     cd rag_systems/simple_rag
     python main.py --query "what is the main cause of climate change?"
     ```
   - 带评估：  
     ```bash
     python main.py --query "what is the main cause of climate change?" --evaluate
     ```
   - 指定 PDF：  
     ```bash
     python main.py --path data/你的文档.pdf --query "你的问题"
     ```

5. **从项目根目录以模块方式运行**（需将 `rag_systems` 置于 Python 路径）：  
   ```bash
   python -m rag_systems.simple_rag.main --path rag_systems/simple_rag/data/Understanding_Climate_Change.pdf --query "..." --evaluate
   ```  
   若从根目录运行且希望默认用本包 data：  
   ```bash
   python -m rag_systems.simple_rag.main --path rag_systems/simple_rag/data/Understanding_Climate_Change.pdf
   ```

## 与原有脚本的关系

- **不修改** `all_rag_techniques_runnable_scripts/simple_rag.py` 及项目根目录的 `helper_functions.py`、`evaluation/evalute_rag.py`。
- 本副本自带 `lib/helpers.py` 与 `lib/evaluation.py`，仅包含 simple_rag 与评估所需的最小逻辑，无项目根依赖。
- 数据与结果路径均在本目录的 `config.py` 中配置，默认使用 `rag_systems/simple_rag/data/`。
