from data_processing.document_loader import *
from data_processing.embedding_generator import *
from data_processing.text_splitter import *
from database.vector_db import *
from llm.model_provider import *
from llm.prompt_engineer import *

if __name__ == "__main__":
    # 构建 HNSW 数据库
    index, chunks = build_hnsw_database("./RAGBase.txt", chunk_strategy="recursive", max_chunk_size=100)
    
    # 进入查询循环
    while True:
        # 读取用户输入
        query1 = input("请输入查询内容（输入 'exit' 退出）：")
        
        # 如果输入 'exit'，退出程序
        if query1 == "exit":
            print("程序已退出。")
            break
        
        result = send_query_to_llm_1(query=query1)

        # 检索相关文档
        related_docs = search_related_chunks(result, index, chunks, k=3)
        related_doc_texts = [doc[0] for doc in related_docs]
        print("与查询最相关的文档原文：")
        for doc, similarity in related_docs:
            print(f"相似度: {similarity:.4f}\n文档: {doc}")
        print("-" * 50)  # 分隔线

        result2 = send_query_to_llm_2(query=result, related_doc_texts=related_doc_texts)