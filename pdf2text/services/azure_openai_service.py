from dataclasses import dataclass
from logging import Logger
from typing import Any, Callable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from lagom.environment import Env
from openai import AsyncAzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)

from pdf2text.models.llm_response import LLMResponse
from pdf2text.protocols.i_azure_openai_service import (
    IAzureOpenAIService,
)
from pdf2text.protocols.i_openai_content_evaluator import IOpenAIContentEvaluator


class AzureOpenAIServiceEnv(Env):
    azure_openai_endpoint: str
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str
    azure_openai_deployed_model_name: str


@dataclass
class AzureOpenAIService(IAzureOpenAIService):
    """
    Azure OpenAI Service implementation.
    """

    env: AzureOpenAIServiceEnv
    content_safety_eval: IOpenAIContentEvaluator
    logger: Logger

    def __post_init__(self) -> None:
        self.client = self.get_client()

    def get_openai_auth_key(self) -> dict[str, str | Callable[[], str]]:
        if self.env.azure_openai_api_key:
            return {"api_key": self.env.azure_openai_api_key}

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        return {"azure_ad_token_provider": token_provider}

    def get_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=self.env.azure_openai_endpoint,
            api_version=self.env.azure_openai_api_version,
            **self.get_openai_auth_key(),  # type: ignore
        )

    def get_deployed_model_name(self) -> str:
        return self.env.azure_openai_deployed_model_name

    def collection_results(
        self, responses: ChatCompletion, num_generations: int
    ) -> list[LLMResponse]:
        self.content_safety_eval.content_safety_check(responses)

        usages = responses.usage.model_dump() if responses.usage else {}
        usages = {k: v for k, v in usages.items() if isinstance(v, int)}
        usages["completion_tokens"] = int(
            usages.get("completion_tokens", 0) / num_generations
        )

        results = []
        for choice in responses.choices:
            results.append(
                LLMResponse(
                    content=choice.message.content if choice.message else "",
                    finish_reason=choice.finish_reason,
                    usages=usages,
                )
            )
        return results

    async def chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 1.0,
        num_generations: int = 1,
    ) -> list[LLMResponse]:
        self.logger.debug("[BEGIN] chat_completion")

        self.client = self.get_client()
        response = await self.client.chat.completions.create(
            model=self.env.azure_openai_deployed_model_name,
            messages=messages,
            temperature=temperature,
            n=num_generations,
        )

        self.logger.debug("[COMPLETED] chat_completion")
        return self.collection_results(response, num_generations)

    async def chat_completion_with_format(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: Any,
        temperature: float = 1.0,
        num_generations: int = 1,
    ) -> list[LLMResponse]:
        self.logger.debug("[BEGIN] chat_completion_with_format")

        self.client = self.get_client()
        responses = await self.client.chat.completions.parse(
            model=self.env.azure_openai_deployed_model_name,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            n=num_generations,
        )

        self.logger.debug("[COMPLETED] chat_completion_with_format")
        return self.collection_results(responses, num_generations)
