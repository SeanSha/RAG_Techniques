"""
Simple RAG 所需的最小辅助函数：文档编码、检索、上下文展示与基于上下文的问答链。
仅包含 simple_rag 主流程与评估流程依赖的接口，不依赖项目根目录的 helper_functions。
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate


def _replace_t_with_space(list_of_documents):
    """将每个文档的 page_content 中的制表符替换为空格。"""
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    return list_of_documents


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    将 PDF 编码为向量库（OpenAI embeddings + FAISS）。

    Args:
        path: PDF 文件路径。
        chunk_size: 每块字符数。
        chunk_overlap: 块间重叠字符数。

    Returns:
        FAISS 向量库。
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = _replace_t_with_space(texts)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    根据问题从 retriever 检索相关文档，返回内容列表（字符串列表）。
    """
    docs = chunks_query_retriever.invoke(question)
    return [doc.page_content for doc in docs]


def show_context(context):
    """打印检索到的上下文列表。"""
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


class QuestionAnswerFromContext(BaseModel):
    """基于上下文生成回答的结构化输出模型。"""
    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


def create_question_answer_from_context_chain(llm):
    """创建「基于上下文回答问题」的 LangChain 链（结构化输出）。"""
    prompt = PromptTemplate(
        template=""" 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """,
        input_variables=["context", "question"],
    )
    chain = prompt | llm.with_structured_output(
        QuestionAnswerFromContext, method="function_calling"
    )
    return chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    根据给定上下文回答问题。

    Args:
        question: 问题字符串。
        context: 上下文字符串（或可 join 的列表，调用方会先 join）。
        question_answer_from_context_chain: 问答链。

    Returns:
        {"answer": str, "context": ..., "question": str}
    """
    if isinstance(context, list):
        context = "\n\n".join(context)
    print("Answering the question from the retrieved context...")
    output = question_answer_from_context_chain.invoke(
        {"question": question, "context": context}
    )
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}
