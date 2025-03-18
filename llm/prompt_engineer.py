
from llm.model_provider import send_request_to_llm

def send_query_to_llm_1(query):
    prompt = "你是中方的谈判专家，当前中印对话内容如下：" + query + "。请你进行总结，请你分析出当前对方的心理状态以及主要矛盾点，根据当前背景给出你的分析结论，用一段连续的话说明。"
    
    url = "http://localhost:11535/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen:14b",
        "prompt": prompt,
        "temperature": 0.8
    }
    result = send_request_to_llm(url, headers, data)
    return result


def send_query_to_llm_2(query, related_doc_texts):
    prompt2 = "你现在作为中方，当前问题为：" + query + "，与该问题相关的文档内容如下：" + " ".join(related_doc_texts) + "。请你以第一人称的方式，直接给出回答。"
    url = "http://localhost:11535/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen:14b",
        "prompt": prompt2,
        "temperature": 0.8
    }

    result = send_request_to_llm(url, headers, data)
    return result