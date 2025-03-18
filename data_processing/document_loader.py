

# 读取文档
def read_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()  # 直接读取整个文档内容
    return content