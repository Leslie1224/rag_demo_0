import os

# 设置 Hugging Face 的镜像站点
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer

# model_path = "./stsb/models"  # Hugging Face 模型名称
model_path = "./models"  # Hugging Face 模型名称
model = SentenceTransformer(model_path)  # 在程序开始时加载模型

# 向量化分块
def vectorize_chunks(chunks):
    chunk_vectors = model.encode(chunks, convert_to_numpy=True).astype('float32')
    return chunk_vectors