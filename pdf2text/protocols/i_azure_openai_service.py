from typing import Any, Protocol

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from pdf2text.models.llm_response import LLMResponse


class IAzureOpenAIService(Protocol):
    def get_client(self) -> AsyncAzureOpenAI:
        """
        Get the Azure OpenAI client.

        :return: An instance of AsyncAzureOpenAI.
        """
        ...

    def get_deployed_model_name(self) -> str:
        """
        Get the name of the deployed model.

        :return: The name of the deployed model.
        """
        ...

    async def chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 1.0,
        num_generations: int = 1,
    ) -> list[LLMResponse]:
        """
        Perform a chat completion using the Azure OpenAI client.

        :param messages: The messages to send in the chat completion.
        :param temperature: The temperature for the completion.
        :param num_generations: The number of generations to produce.
        :return: The content of the response message.
        """
        ...

    async def chat_completion_with_format(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: Any,
        temperature: float = 1.0,
        num_generations: int = 1,
    ) -> list[LLMResponse]:
        """
        Perform a chat completion and parse the response into the specified format.

        :param messages: The messages to send in the chat completion.
        :param response_format: The format to parse the response into.
        :param temperature: The temperature for the completion.
        :param num_generations: The number of generations to produce.
        :return: The parsed response.
        """
        ...
