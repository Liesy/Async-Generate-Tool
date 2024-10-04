import time
from asynic_generate import LanguageModel

api_config = {
    "openai": {
        "base_url": "",
        "api_key": "",
    },
    "anthropic": {
        "base_url": "",
        "api_key": "",
    },
    "Llama-3.1-8B-Instruct": {"base_url": "http://0.0.0.0:8000/v1", "api_key": "EMPTY"},
    "Qwen2-7B-Instruct": {"base_url": "http://0.0.0.0:8006/v1", "api_key": "EMPTY"},
}

# llm = asynic_generate.LanguageModel("claude-3-haiku-20240307", api_config)
llm = LanguageModel("gpt-4o-mini-2024-07-18", api_config)


# * single prompt
text = "what's your name"
start = time.time()
ret, history = llm.get_response(text)
during = time.time() - start
print(f"single generation in {during} seconds:\n{ret}")


# * prompt batch
texts = ["what's your name", "what's the weather like in Beijing", "hello"]
start = time.time()
rets, historys = llm.get_response(texts)
during = time.time() - start
print(f"batch generation in {during} seconds:")
for r in rets:
    print(r)
