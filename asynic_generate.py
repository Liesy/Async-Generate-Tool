import asyncio, random, copy
import openai, anthropic


class GPT:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 10
    API_TIMEOUT = 20

    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        conv: list[dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.API_TIMEOUT,
                )
                output = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e, conv)
                await asyncio.sleep(random.uniform(1, self.API_RETRY_SLEEP))

            await asyncio.sleep(self.API_QUERY_SLEEP)
        return output

    async def batched_generate(
        self,
        convs_list: list[list[dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        coroutines = [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]
        outputs = await asyncio.gather(*coroutines)
        return outputs


class Claude:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 10
    API_TIMEOUT = 20

    def __init__(self, model_name, api_key, base_url) -> None:
        self.model_name = model_name
        self.model = anthropic.AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
        )

    async def generate(
        self,
        conv: list[dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Args:
            conv: List of conversations
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = await self.model.messages.create(
                    model=self.model_name,
                    max_tokens=max_n_tokens,
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.API_TIMEOUT,
                )
                output = completion.content[0].text
                break
            except anthropic.APIError as e:
                print(type(e), e, conv)
                await asyncio.sleep(random.uniform(1, self.API_RETRY_SLEEP))

            await asyncio.sleep(self.API_QUERY_SLEEP)
        return output

    async def batched_generate(
        self,
        convs_list: list[list[dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        coroutines = [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]
        outputs = await asyncio.gather(*coroutines)
        return outputs


class vLLM:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 10
    API_TIMEOUT = 20

    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        conv: list[dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.API_TIMEOUT,
                )
                output = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e, conv)
                await asyncio.sleep(random.uniform(1, self.API_RETRY_SLEEP))

            await asyncio.sleep(self.API_QUERY_SLEEP)
        return output

    async def batched_generate(
        self,
        convs_list: list[list[dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        coroutines = [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]
        outputs = await asyncio.gather(*coroutines)
        return outputs


class LanguageModel:
    USR = "user"
    BOT = "assistant"

    def __init__(
        self,
        model_name: str,
        api_config: dict,
        system_prompt: str = "You are a helpful AI assistant",
    ):
        self.model_name = model_name
        self.api_config = api_config
        self.system_prompt = system_prompt

        self.__model = self.__load_model()
        self.chat_template = [{"role": "system", "content": self.system_prompt}]

    def __load_model(self):
        if "gpt" in self.model_name.lower():
            lm = GPT(
                self.model_name,
                self.api_config["openai"]["api_key"],
                self.api_config["openai"]["base_url"],
            )
        elif "claude" in self.model_name.lower():
            lm = Claude(
                self.model_name,
                self.api_config["anthropic"]["api_key"],
                self.api_config["anthropic"]["base_url"],
            )
        elif (
            "llama" in self.model_name.lower()
            or "vicuna" in self.model_name.lower()
            or "wizard" in self.model_name.lower()
            or "qwen" in self.model_name.lower()
        ):
            lm = vLLM(
                self.model_name,
                self.api_config[self.model_name]["api_key"],
                self.api_config[self.model_name]["base_url"],
            )
        else:
            raise ValueError(f"Error in loading model {self.model_name}")
        return lm

    def get_response(
        self,
        query: str | list[str],
        history: list[dict[str, str]] | None = None,
        max_n_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> tuple[str | list[str], list[dict[str, str]]]:

        self.ret_output, self.ret_history = None, None
        history = copy.deepcopy(self.chat_template) if history is None else history

        async def str_entrance():
            prompt = copy.deepcopy(history)
            prompt.append({"role": self.USR, "content": query})
            output = await self.__model.generate(
                prompt, max_n_tokens, temperature, top_p
            )
            self.ret_output = output
            self.ret_history = prompt.append({"role": self.BOT, "content": output})

        async def list_entrance():
            prompt_list = [copy.deepcopy(history) for _ in range(len(query))]
            for p, q in zip(prompt_list, query):
                p.append({"role": self.USR, "content": q})
            outputs = await self.__model.batched_generate(
                prompt_list, max_n_tokens, temperature, top_p
            )
            for p, r in zip(prompt_list, outputs):
                p.append({"role": self.BOT, "content": r})
            self.ret_output = outputs
            self.ret_history = prompt_list

        if isinstance(query, str):
            asyncio.run(str_entrance())
        else:
            asyncio.run(list_entrance())

        return self.ret_output, self.ret_history
