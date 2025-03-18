

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
