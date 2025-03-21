import faiss
from data_processing.embedding_generator import model, vectorize_chunks
from data_processing.document_loader import *
from data_processing.text_splitter import *
from calculate_time import *

# 创建 HNSW 索引
def create_hnsw_index(doc_vectors, M=16, efConstruction=200):
    d = doc_vectors.shape[1]  # 向量维度
    faiss.normalize_L2(doc_vectors)  # 归一化向量
    index = faiss.IndexHNSWFlat(d, M)  # 使用 HNSW 索引
    index.hnsw.efConstruction = efConstruction  # 设置构建参数
    index.metric_type = faiss.METRIC_INNER_PRODUCT  # 设置度量方式为内积
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
        
    # 归一化查询向量
    faiss.normalize_L2(query_vector)
    
    # 设置搜索参数
    index.hnsw.efSearch = efSearch
    
    # 检索
    distances, indices = index.search(query_vector, k)  # 先检索前 k 个结果
    
    # 将内积距离转换为余弦相似度（Faiss 返回的是内积，越大表示越相似）
    similarities = (1 + distances) / 2  # 将内积归一化到 [0, 1] 范围
    
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

    startTime = record_timestamp()
    # 读取文档
    content = read_documents(file_path)
    endTime = record_timestamp()
    read_documents_time = calculate_duration(startTime, endTime)
    print("文档读取时间:", read_documents_time)
    print("文档读取完成")
    
    startTime = record_timestamp()
    # 分块知识库
    chunks = chunk_knowledge_base(content, chunk_strategy, max_chunk_size)
    endTime = record_timestamp()
    chunk_knowledge_time = calculate_duration(startTime, endTime)
    print("知识库分块时间:", chunk_knowledge_time)
    print(f"知识库分块完成，共生成 {len(chunks)} 个分块")
    
    startTime = record_timestamp()
    # 向量化分块
    chunk_vectors = vectorize_chunks(chunks)
    endTime = record_timestamp()
    vectorize_time = calculate_duration(startTime, endTime)
    print("分块向量化时间:", vectorize_time)
    print("分块向量化完成")
    
    startTime = record_timestamp()
    # 创建并保存 HNSW 索引
    index = create_hnsw_index(chunk_vectors)
    endTime = record_timestamp()
    create_hnsw_index_time = calculate_duration(startTime, endTime)
    print("创建 HNSW 索引时间:", create_hnsw_index_time)
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
