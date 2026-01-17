import os
import json
import time
from abc import ABC, abstractmethod

import anthropic
import tiktoken

import openai


def num_tokens_from_messages(message, model='gpt-3.5-turbo-0301'):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]['content']))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_anthropic_config(
    message: str | list,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = 'You are a helpful assistant.',
    model: str = 'claude-2.1',
    tools: list = None,
) -> dict:
    if isinstance(message, list):
        config = {
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'messages': message,
        }
    else:
        config = {
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'messages': [
                {'role': 'user', 'content': [{'type': 'text', 'text': message}]},
            ],
        }

    if tools:
        config['tools'] = tools

    return config


def request_anthropic_engine(
    config,
    max_retries=40,
    timeout=500,
    prompt_cache=False,
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config['messages'][0]['content'][0]['cache_control'] = {
                    'type': 'ephemeral',
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception:
            print('Unknown error. Waiting...', exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                print('Request timed out. Retrying...')
            else:
                print('Retrying after an unknown error...')
            time.sleep(10 * retries)
        retries += 1

    return ret


def create_chatgpt_config(
    message: str | list,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = 'You are a helpful assistant.',
    model: str = 'gpt-3.5-turbo',
) -> dict:
    if isinstance(message, list):
        config = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'n': batch_size,
            'messages': [{'role': 'system', 'content': system_message}] + message,
        }
    else:
        config = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'n': batch_size,
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': message},
            ],
        }
    return config


def create_chatgpt_config_agent(
    message: str | list,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    model: str = 'gpt-3.5-turbo',
) -> dict:
    if isinstance(message, list):
        config = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'n': batch_size,
            'messages': message,
            'tools': tools,
            'tool_choice': 'auto',
        }
    else:
        config = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'n': batch_size,
            'messages': [
                {'role': 'user', 'content': message},
            ],
            'tools': tools,
            'tool_choice': 'auto',
        }
    return config


def create_chatgpt_config_claude_think(
    message: str | list,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    model: str = 'claude-3-7-sonnet-20250219-thinking',
) -> dict:
    if isinstance(message, list):
        config = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'n': batch_size,
            'messages': message,
            'extra_body': {
                'thinking': {
                    'type': 'enabled',
                    'budget_tokens': 16384,
                },
            },
        }
    else:
        config = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'n': batch_size,
            'messages': [
                {'role': 'user', 'content': message},
            ],
            'extra_body': {
                'thinking': {
                    'type': 'enabled',
                    'budget_tokens': 16384,
                },
            },
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception('end of time')


def request_chatgpt_engine(config, base_url=None, max_retries=20, timeout=1200, api_key=None):
    ret = None
    retries = 0

    if api_key is not None:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            ret = client.chat.completions.create(**config, timeout=timeout)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                print(e)
                if 'maximum context length' in str(e) or 'maximum length' in str(e) or 'max_prompt_tokens' in str(e):
                    return None
                continue  # retry for R1 model
                raise Exception('Invalid API Request')
            if isinstance(e, openai.RateLimitError):
                print('Rate limit exceeded. Waiting...')
                print(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print('API connection error. Waiting...')
                print(e)
                time.sleep(5)
            else:
                print('Unknown error. Waiting...')
                print(e)
                time.sleep(1)

        retries += 1

    return ret


def request_chatgpt_engine_rl(config, base_url=None, max_retries=3, timeout=500, api_key=None):
    ret = None
    retries = 0

    if api_key is not None:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            ret = client.chat.completions.create(**config, timeout=timeout)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                print(e)
                raise Exception('Invalid API Request')
            if isinstance(e, openai.RateLimitError):
                print('Rate limit exceeded. Waiting...')
                print(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print('API connection error. Waiting...')
                print(e)
                time.sleep(5)
            elif isinstance(e, openai.Timeout):
                print('Request timed out. Retrying...')
                print(e)
                time.sleep(5)
            else:
                print('Unknown error. Waiting...')
                print(e)
                time.sleep(1)

        retries += 1

    return ret


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(
        self,
        message: str,
        num_samples: int = 1,
        prompt_cache: bool = False,
    ) -> list[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class AnthropicChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    _STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    _USER_REPLY_EDIT_MESSAGE = """File is successfully edited"""

    tools = [
        {
            'name': 'str_replace_editor',
            'description': _STR_REPLACE_EDITOR_DESCRIPTION,
            'input_schema': {
                'type': 'object',
                'properties': {
                    'path': {
                        'description': 'Full path to file, e.g. `folder/file.py`.',
                        'type': 'string',
                    },
                    'old_str': {
                        'description': 'Required parameter containing the string in `path` to replace.',
                        'type': 'string',
                    },
                    'new_str': {
                        'description': 'Optional parameter containing the new string (if not given, no string will be added).',
                        'type': 'string',
                    },
                },
                'required': ['path', 'old_str'],
            },
        },
    ]

    MAX_CODEGEN_ITERATIONS = 10

    # specialized codegen with tool
    def codegen_w_tool(
        self,
        message: str | list,
        num_samples: int = 1,
        prompt_cache: bool = False,
    ) -> list[dict]:
        def _build_response_and_extract(response, messages, iter):
            json_response = response.to_dict()

            contains_tool = False
            # formulate the messages
            json_response.pop('id')
            json_response.pop('model')
            json_response.pop('stop_reason')
            json_response.pop('stop_sequence')
            json_response.pop('type')
            json_response.pop('usage')

            messages.append(json_response)

            response_content = []

            for json_message in json_response['content']:
                if json_message['type'] == 'tool_use':
                    contains_tool = True
                    # each tool use requires a response
                    response_content.append(
                        {
                            'type': 'tool_result',
                            'tool_use_id': json_message['id'],
                            'content': self._USER_REPLY_EDIT_MESSAGE,
                        },
                    )

            if contains_tool:
                messages.append(
                    {
                        'role': 'user',
                        'content': response_content,
                    },
                )
            else:
                if iter == 0:
                    # if the first iteration does not contain the tool, likely the model is doing some CoT for debugging
                    # append encouraging message
                    messages.append(
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': 'Please generate editing commands to fix the issue',
                                },
                            ],
                        },
                    )
                    contains_tool = True

            return messages, contains_tool

        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            self.logger.info(' === Generating ====')
            # initialized the traj
            traj = {
                'response': [],
                'usage': {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'cache_creation_token': 0,
                    'cache_read_input_tokens': 0,
                },
            }

            # create the initial config and messages
            messages = [
                {'role': 'user', 'content': [{'type': 'text', 'text': message}]},
            ]

            for iteration in range(self.MAX_CODEGEN_ITERATIONS):
                config = create_anthropic_config(
                    message=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=1,
                    model=self.name,
                    tools=self.tools,
                )
                ret = request_anthropic_engine(
                    config,
                    self.logger,
                    prompt_cache=True,  # prompt cache should be always true as we at least should query twice
                )

                if ret:
                    # add the response to the traj
                    traj['response'].append([reply.to_dict() for reply in ret.content])

                    # pretty dump the response
                    for reply in ret.content:
                        self.logger.info(json.dumps(reply.to_dict(), indent=2))

                    # update the usage
                    traj['usage']['completion_tokens'] += ret.usage.output_tokens
                    traj['usage']['prompt_tokens'] += ret.usage.input_tokens
                    traj['usage']['cache_creation_token'] += ret.usage.cache_creation_input_tokens
                    traj['usage']['cache_read_input_tokens'] += ret.usage.cache_read_input_tokens

                    messages, contains_tool = _build_response_and_extract(
                        ret,
                        messages,
                        iteration,
                    )

                    if not contains_tool:
                        break
                else:
                    assert False, 'No response from the engine'  # this should not happen

            if ret:
                trajs.append(traj)
            else:
                trajs.append(
                    {
                        'response': '',
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                        },
                    },
                )

        return trajs

    def codegen(
        self,
        message: str | list,
        num_samples: int = 1,
        prompt_cache: bool = False,
    ) -> list[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_anthropic_engine(
                config,
                self.logger,
                prompt_cache=prompt_cache,
            )

            if ret:
                trajs.append(
                    {
                        'response': ret.content[0].text,
                        'usage': {
                            'completion_tokens': ret.usage.output_tokens,
                            'prompt_tokens': ret.usage.input_tokens,
                            'cache_creation_token': 0
                            if not prompt_cache
                            else ret.usage.cache_creation_input_tokens,
                            'cache_read_input_tokens': 0
                            if not prompt_cache
                            else ret.usage.cache_read_input_tokens,
                        },
                    },
                )
            else:
                trajs.append(
                    {
                        'response': '',
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                        },
                    },
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def codegen(
        self,
        message: str,
        num_samples: int = 1,
        prompt_cache: bool = False,
    ) -> list[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(
                config,
                base_url=os.getenv("OPENAI_BASE_URL", f'http://127.0.0.1:8000/v1'),
                api_key=os.getenv("OPENAI_API_KEY", 'EMPTY'),
                )
        if ret:
            responses = [choice.message.content if not getattr(choice, "reasoning_content", "") else f"<think>{choice.message.reasoning_content}</think>{choice.message.content}" for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = ['']
            completion_tokens = 0
            prompt_tokens = 0

        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                'response': responses[0],
                'usage': {
                    'completion_tokens': completion_tokens,
                    'prompt_tokens': prompt_tokens,
                },
            },
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    'response': response,
                    'usage': {
                        'completion_tokens': 0,
                        'prompt_tokens': 0,
                    },
                },
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def codegen(
        self,
        message: str,
        num_samples: int = 1,
        prompt_cache: bool = False,
    ) -> list[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config,
                base_url='https://api.deepseek.com',
            )
            if ret:
                trajs.append(
                    {
                        'response': ret.choices[0].message.content,
                        'usage': {
                            'completion_tokens': ret.usage.completion_tokens,
                            'prompt_tokens': ret.usage.prompt_tokens,
                        },
                    },
                )
            else:
                trajs.append(
                    {
                        'response': '',
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                        },
                    },
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class KimiDevChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def codegen(
        self,
        message: str | list,
        num_samples: int = 1,
        prompt_cache: bool = False,
    ) -> list[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            # print(config)
            # ************* make sure the response exits!!!*******************
            ret_len = 0
            while ret_len == 0:
                ret = request_chatgpt_engine(
                    config,
                    base_url=os.getenv("OPENAI_BASE_URL", f'http://127.0.0.1:8000/v1'),
                    api_key=os.getenv("OPENAI_API_KEY", 'EMPTY'),
                )
                ret_len = len(ret.choices)
            # ****************************************************************
            if ret:
                trajs.append(
                    {
                        'response': ret.choices[0].message.content,
                        'usage': {
                            'completion_tokens': ret.usage.completion_tokens,
                            'prompt_tokens': ret.usage.prompt_tokens,
                        },
                    },
                )
            else:
                trajs.append(
                    {
                        'response': '',
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                        },
                    },
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False




def make_model(
    model: str,
    backend: str,
    batch_size: int = 1,
    max_tokens: int = 1024 * 16,
    temperature: float = 0.0,
):
    if backend == 'openai':
        return OpenAIChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    if backend == 'anthropic':
        return AnthropicChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    if backend == 'deepseek':
        return DeepSeekChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    if backend == 'deepseekr1':
        return DeepSeekR1ChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    if backend == 'kimidev':
        return KimiDevChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    raise NotImplementedError
