

import requests
import json

def send_request_to_llm(url, headers, data):
    try:
        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=data)
        # 检查请求是否成功，如果不成功会抛出异常
        response.raise_for_status()

        # 初始化一个空字符串，用于存储最终拼接的结果
        combined_response = ""
        # 将响应内容按行分割，得到一个包含每行内容的列表
        lines = response.text.splitlines()
        for line in lines:
            try:
                # 尝试将每行内容解析为 JSON 对象
                json_obj = json.loads(line)
                # 从解析后的 JSON 对象中提取 response 字段的值
                combined_response += json_obj["response"]
            except json.JSONDecodeError:
                # 如果解析失败，打印提示信息
                print(f"无法解析行: {line}")


        # print("-" * 50)  # 分隔线 
        # 打印拼接后的完整内容
        print(combined_response)
        # print("-" * 50)  # 分隔线
        return combined_response

    except requests.exceptions.HTTPError as http_err:
        # 处理 HTTP 请求错误
        print(f"HTTP 错误发生: {http_err}")
    except Exception as err:
        # 处理其他异常
        print(f"发生其他错误: {err}")
