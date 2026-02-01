"""
Simple RAG Local：完全本地运行。
- 向量：HuggingFaceEmbeddings（sentence-transformers），无 API 调用。
- 问答链：接受任意 LangChain LLM（main 中传入 ChatOllama），使用普通文本输出（无 function_calling）。
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 本包 config，延迟导入避免循环
def _get_embedding_model():
    import config as _config
    return _config.EMBEDDING_MODEL


def _replace_t_with_space(list_of_documents):
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    return list_of_documents


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    将 PDF 编码为向量库（本地 HuggingFace embeddings + FAISS），不调用任何云端 API。
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = _replace_t_with_space(texts)
    model_name = _get_embedding_model()
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    docs = chunks_query_retriever.invoke(question)
    return [doc.page_content for doc in docs]


def show_context(context):
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


def create_question_answer_from_context_chain(llm):
    """
    创建「基于上下文回答问题」的链。使用普通文本输出（兼容 Ollama 等本地模型）。
    """
    prompt = PromptTemplate(
        template=""" 
Based ONLY on the following context, answer the question in one short paragraph. Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer (only the answer, no preamble):""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    if isinstance(context, list):
        context = "\n\n".join(context)
    import sys
    print("正在调用本地 Ollama 生成回答（首次可能需 30 秒～2 分钟，请稍候）...", flush=True)
    input_data = {"question": question, "context": context}
    # 流式输出：边生成边打印，避免长时间无反馈
    full = ""
    try:
        for chunk in question_answer_from_context_chain.stream(input_data):
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, dict):
                text = chunk.get("content") or chunk.get("output") or chunk.get("text") or ""
                for v in chunk.values():
                    if isinstance(v, str):
                        text = v
                        break
            else:
                text = str(chunk)
            if text:
                full += text
                print(text, end="", flush=True)
        print(flush=True)  # 流式输出后换行
    except Exception:
        # 若 stream 不可用则回退到 invoke
        result = question_answer_from_context_chain.invoke(input_data)
        full = result.strip() if isinstance(result, str) else getattr(result, "content", str(result))
    answer = full.strip() if full else ""
    return {"answer": answer, "context": context, "question": question}
