import faiss
import numpy as np
import os
# 设置 Hugging Face 的镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer

# 初始化模型
model_path = "./stsb/models"  # Hugging Face 模型名称
model = SentenceTransformer(model_path)  # 在程序开始时加载模型

# 读取文档
def read_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()  # 直接读取整个文档内容
    return content

# 递归分块函数
def recursive_chunking(text, max_chunk_size=100, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", "。", "，", " "]  # 默认分隔符
    
    chunks = []
    current_chunk = ""
    
    # 如果没有分隔符，直接返回整个文本
    if not separators:
        chunks.append(text)
        return chunks
    
    # 使用第一个分隔符进行分割
    separator = separators[0]
    parts = text.split(separator)
    
    for part in parts:
        # 如果当前块加上新部分的大小超过最大块大小，则递归分割
        if len(current_chunk) + len(part) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            chunks.extend(recursive_chunking(part, max_chunk_size, separators[1:]))
            current_chunk = ""
        else:
            current_chunk += part + separator if separator else part
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# 分块知识库
def chunk_knowledge_base(content, chunk_strategy="recursive", max_chunk_size=100):
    chunks = []
    if chunk_strategy == "recursive":
        chunks = recursive_chunking(content, max_chunk_size)
    return chunks

# 向量化分块
def vectorize_chunks(chunks):
    chunk_vectors = model.encode(chunks, convert_to_numpy=True).astype('float32')
    return chunk_vectors

# 创建 HNSW 索引
def create_hnsw_index(doc_vectors, M=16, efConstruction=200):
    d = doc_vectors.shape[1]  # 向量维度
    index = faiss.IndexHNSWFlat(d, M)  # 使用 HNSW 索引
    index.hnsw.efConstruction = efConstruction  # 设置构建参数
    index.add(doc_vectors)  # 添加向量到索引
    return index

# 保存索引和分块
def save_index_and_chunks(index, chunks, index_path="faiss_index.bin", chunks_path="chunks.txt"):
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")

# 加载索引和分块
def load_index_and_chunks(index_path="faiss_index.bin", chunks_path="chunks.txt"):
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]
    return index, chunks

# 检索相似分块
def search_related_chunks(query, index, chunks, k=3, efSearch=100, similarity_threshold=0.00):
    query_vector = model.encode([query], convert_to_numpy=True).astype('float32')  # 使用全局的 model
    
    # 设置搜索参数
    index.hnsw.efSearch = efSearch
    
    # 检索
    distances, indices = index.search(query_vector, k)  # 先检索前 k 个结果
    
    # 将距离转换为相似度（Faiss 返回的是 L2 距离，越小表示越相似）
    similarities = 1 / (1 + distances)  # 将距离转换为相似度
    
    # 筛选满足相似度阈值的结果
    filtered_results = []
    for i in range(len(indices[0])):
        if similarities[0][i] >= similarity_threshold:
            filtered_results.append((chunks[indices[0][i]], similarities[0][i]))
    
    # 如果满足阈值的结果超过 3 个，只返回前 3 个
    if len(filtered_results) > 3:
        filtered_results = filtered_results[:3]
    
    # 返回结果（包含文本和相似度）
    return filtered_results

# 主函数：构建 HNSW 数据库
def build_hnsw_database(file_path, chunk_strategy="recursive", max_chunk_size=100):
    # 读取文档
    content = read_documents(file_path)
    print("文档读取完成")
    
    # 分块知识库
    chunks = chunk_knowledge_base(content, chunk_strategy, max_chunk_size)
    print(f"知识库分块完成，共生成 {len(chunks)} 个分块")
    
    # 向量化分块
    chunk_vectors = vectorize_chunks(chunks)
    print("分块向量化完成")
    
    # 创建并保存 HNSW 索引
    index = create_hnsw_index(chunk_vectors)
    save_index_and_chunks(index, chunks)
    print("HNSW 数据库已创建并保存")
    return index, chunks

# 主函数：加载 HNSW 数据库并检索相关文档
def search_in_hnsw_database(query, k=2):
    # 加载 HNSW 索引
    index, chunks = load_index_and_chunks()
    # 检索相关文档
    related_chunks = search_related_chunks(query, index, chunks, k)
    return related_chunks

if __name__ == "__main__":
    # 构建 HNSW 数据库
    index, chunks = build_hnsw_database("./RAGBase.txt", chunk_strategy="recursive", max_chunk_size=100)
    
    # 进入查询循环
    while True:
        # 读取用户输入
        query = input("请输入查询内容（输入 'exit' 退出）：")
        
        # 如果输入 'exit'，退出程序
        if query == "exit":
            print("程序已退出。")
            break
        
        # 检索相关文档
        related_docs = search_related_chunks(query, index, chunks, k=3)
        print("与查询最相关的文档原文：")
        for doc, similarity in related_docs:
            print(f"相似度: {similarity:.4f}\n文档: {doc}")
        print("-" * 50)  # 分隔线