from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict

# 导入原有模块
from data_processing.document_loader import *
from data_processing.text_splitter import *
from database.vector_db import *
from llm.model_provider import *
from llm.prompt_engineer import *
from calculate_time import *

app = FastAPI(
    title="RAG API Service",
    description="基于FastAPI封装的RAG问答系统接口",
    version="1.0.0"
)

# 必须第一个添加中间件（重要！）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.1.87:5173"],  # 精确指定前端地址
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # 明确允许的方法
    allow_headers=["*"],
    expose_headers=["*"]
)


# 全局存储索引和分块
global_index = None  
global_chunks = None

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3
    similarity_threshold: float = 0.6

class DocumentResponse(BaseModel):
    content: str
    similarity: float

class QueryResponse(BaseModel):
    initial_answer: str
    related_documents: List[DocumentResponse]
    final_answer: str

@app.on_event("startup")
async def initialize_hnsw():
    """服务启动时自动初始化HNSW数据库"""
    global global_index, global_chunks
    try:
        global_index, global_chunks = build_hnsw_database(
            "./RAGBase.txt", 
            chunk_strategy="recursive",
            max_chunk_size=300
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"数据库初始化失败: {str(e)}"
        )
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    处理用户查询的RAG流程
    
    Parameters:
    - text: 查询文本
    - top_k: 返回相关文档数量（默认3）
    - similarity_threshold: 相似度阈值（默认0.6）
    """
    if not global_index:
        raise HTTPException(
            status_code=503, 
            detail="数据库未初始化完成，请稍后重试"
        )
    
    try:
        # 第一阶段LLM处理
        # initial_result = send_query_to_llm_1(query=request.text)
        summary = request.text
        
        # 检索相关文档
        related_docs = search_related_chunks(
            # initial_result, 
            summary,
            global_index, 
            global_chunks, 
            k=request.top_k
        )
        print(f" 相似度 {related_docs}")
        
        # 过滤低于阈值的文档
        filtered_docs = [
            (doc, sim) for doc, sim in related_docs 
            if sim >= request.similarity_threshold
        ]
        
        # 第二阶段LLM处理
        final_answer = send_query_to_llm_2(
            query=summary,
            related_doc_texts=[doc[0] for doc in filtered_docs]
        )
        
        return {
            "initial_answer": summary,
            "related_documents": [
                {"content": doc, "similarity": sim} 
                for doc, sim in filtered_docs
            ],
            "final_answer": final_answer
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"查询处理失败: {str(e)}"
        )

