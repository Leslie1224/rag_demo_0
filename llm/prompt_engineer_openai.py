from openai import OpenAI
import argparse

# 初始化 OpenAI 客户端
base_url = "http://10.72.129.218:18080/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

# 解析命令行参数
parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--stream', action='store_true', help='use stream chat or not (default: False)')
args = parser.parse_args()

# 发送请求到 LLM
def send_request_to_llm(prompt, model="qwen2.5-7b", stream=False):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )
    if stream:
        # 流式输出
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, flush=True, end='')
        print('')
    else:
        # 非流式输出
        return response.choices[0].message.content

# 第一个函数：总结对话内容
def send_query_to_llm_1(query):
    prompt = "你是中方的谈判专家，当前中印对话内容如下：" + query + "。请你进行总结，请你分析出当前对方的心理状态以及主要矛盾点，根据当前背景给出你的分析结论，用一段连续的话说明。"
    return send_request_to_llm(prompt, model="qwen2.5-7b", stream=args.stream)

# 第二个函数：结合相关文档回答问题
def send_query_to_llm_2(query, related_doc_texts):
    prompt = "你现在作为中方，当前问题为：" + query + "，与该问题相关的文档内容如下：" + " ".join(related_doc_texts) + "。请你以第一人称的方式，直接给出回答。"
    return send_request_to_llm(prompt, model="qwen2.5-7b", stream=args.stream)

# # 示例调用
if __name__ == "__main__":
    query = "印方提出边境问题需要重新谈判。"
    related_docs = ["中印边境问题历史悠久，双方曾多次谈判。", "当前边境局势紧张，需要谨慎处理。"]

    print("send_query_to_llm_1 结果：")
    result_1 = send_query_to_llm_1(query)
    print(result_1)

    print("\nsend_query_to_llm_2 结果：")
    result_2 = send_query_to_llm_2(query, related_docs)
    print(result_2)