问题：

关键词检索不出来？（分块策略：递归分块/语义分块方法？和距离查找方法—距离种类和距离归一化的方法？）


embedding好的模型，各方面的优势（参数：维度有影响吗？），分块策略做实验（recrusive），索引类型，rerank模型(调研下现在有哪些，效果如何)

- **词元分割 (Token)**：基于token分割文本，存在几种不同的token测量方式。
- **字符分割 (Character)**：基于用户定义的单个字符分割文本，是一种较为简单的方法。
- **[实验性] 语义块分割 (Semantic Chunker)**：首先按句子分割文本，然后如果相邻句子在语义上足够相似，则将它们合并。此方法源自Greg Kamradt。

- **递归分割 (Recursive)**：基于用户定义的字符列表递归分割文本，旨在保持相关文本片段相邻。推荐作为初始分割方法。


CUDA_VISIBLE_DEVICES=2 ollama run qwen:14b 直接指定该模型运行的gpu，该服务还是会运行在gpu0上而非gpu2

在启动ollama服务的时候就指定： $ CUDA_VISIBLE_DEVICES=2 ollama serve

则该服务会与运行在gpu2上，否则python程序和大模型都是用gpu0，程序很可能因为gpu0显存不足而崩溃