import os
import ollama
from typing import List
import numpy as np

# 配置 Ollama 服务地址（如果不在本地默认地址）
os.environ['OLLAMA_HOST'] = 'http://2.ndsl:11434'

def vectorize_chunks(texts: List[str], embed_model: str, **kwargs) -> np.ndarray:
    """
    使用Ollama生成文本嵌入向量
    
    :param texts: 要向量化的文本列表
    :param embed_model: Ollama使用的嵌入模型名称
    :param kwargs: Ollama客户端配置参数（如host, auth等）
    :return: 形状为(len(texts), 嵌入维度)的numpy数组
    """
    ollama_client = ollama.Client(**kwargs)
    embeddings = []
    
    for text in texts:
        try:
            response = ollama_client.embeddings(model=embed_model, prompt=text)
            # response = ollama_client.embeddings(model=embed_model, prompt=text, batch_size=32)  # 启用批量优化

            embeddings.append(response["embedding"])
        except Exception as e:
            # 建议添加更详细的错误处理逻辑
            raise RuntimeError(f"生成嵌入失败: {str(e)}") from e
    
    return np.array(embeddings, dtype='float32')
# def vectorize_chunks(texts: list[str], embed_model, **kwargs) -> np.ndarray:
#     embed_text = []
#     ollama_client = ollama.Client(**kwargs)
#     for text in texts:
#         data = ollama_client.embeddings(model=embed_model, prompt=text)
#         embed_text.append(data["embedding"])

#     return embed_text

# # 示例用法
if __name__ == "__main__":
    chunks = [
        "这是第一段文本。",
        "这是第二段文本，稍长一些。"
    ]
    vectors = vectorize_chunks(chunks,embed_model="nomic-embed-text", host="http://2.ndsl:11434")
    print(f"向量维度: {vectors.shape}")  # 输出示例: (2, 768)