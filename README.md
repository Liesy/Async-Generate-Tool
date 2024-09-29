# Async Generate Tool for Large Language Models (APIs)

<font color=FF0000>Please Star it if it's useful for you</font>

## Author

- Yang Li, Master's degree in reading
- Institute of Computing Technology, Chinese Academy of Sciences

## Requirements

- [vllm](https://github.com/vllm-project/vllm)
- openai
- anthropic
- tqdm

## Usage

see `example.ipynb` for more details.

## Introduction

最近造大规模数据集，需要借助多个大模型生成。

### 瓶颈

1. 使用不同的API，都要重新编写生成代码，重复造轮子显然不是合格的计算机学生（
2. 大批量数据，如果顺序处理则浪费时间在请求上，用多进程又会在同时处理长文本和短文本时效率低下（长文本达到api的limit，导致同批的短文本也反复请求失败）

### 思路

显然是一个IO密集型任务，用大批量异步生成，可以让长文本的生成不再阻塞短文本，大大提升工作效率

### 方法

晚上心血来潮写了个便捷的小工具

1. 用LanguageModel类，传入模型名称和API的配置，就能直接把字符串扔进去使用里面的get_response函数生成回复，不用再看openai和anthropic的文档写一堆东西了
2. 支持大规模数据的异步生成，效率提高，且避免了多进程可能出现的问题（长文本达到api的limit，导致同批的短文本也反复请求失败）
3. 可以配合vLLM或者sglang使用，开源模型也能cover住

顺手点个star吧🫰🏻