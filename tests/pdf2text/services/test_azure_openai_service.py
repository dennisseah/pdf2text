from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from pdf2text.services.azure_openai_service import AzureOpenAIService


@pytest.fixture
def fn_mock_service(mocker: MockerFixture) -> Callable[[bool], AzureOpenAIService]:
    def wrapper(with_api_key: bool) -> AzureOpenAIService:
        patched_azure_openai = mocker.patch(
            "pdf2text.services.azure_openai_service.AsyncAzureOpenAI",
            autospec=True,
        )

        mock_cred = MagicMock()
        mock_cred.get_token.return_value = MagicMock(
            token="mock_token",
            scope="https://example.com/.default",
        )
        mocker.patch(
            "pdf2text.services.azure_openai_service.DefaultAzureCredential",
            return_value=mock_cred,
        )

        env = MagicMock()
        if not with_api_key:
            env.azure_openai_api_key = None
        svc = AzureOpenAIService(
            env=env, content_safety_eval=MagicMock(), logger=MagicMock()
        )
        patched_azure_openai.assert_called_once()
        return svc

    return wrapper


def test_get_client(
    fn_mock_service: Callable[[bool], AzureOpenAIService], mocker: MockerFixture
):
    mock_service = fn_mock_service(with_api_key=True)  # type: ignore
    assert mock_service.client is not None


def test_get_client_without_api_key(
    fn_mock_service: Callable[[bool], AzureOpenAIService], mocker: MockerFixture
):
    mock_service = fn_mock_service(with_api_key=False)  # type: ignore
    assert mock_service.client is not None


def test_get_deployed_model_name(
    fn_mock_service: Callable[[bool], AzureOpenAIService],
):
    mock_service = fn_mock_service(with_api_key=True)  # type: ignore
    mock_service.env.azure_openai_deployed_model_name = "test-model"

    model_name = mock_service.get_deployed_model_name()
    assert model_name == "test-model"


@pytest.mark.asyncio
async def test_chat_completion(
    fn_mock_service: Callable[[bool], AzureOpenAIService],
):
    mock_service = fn_mock_service(with_api_key=True)  # type: ignore
    mock_service.client.chat.completions = MagicMock()
    mock_service.client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Test response"), finish_reason="stop"
                )
            ],
            usage=None,
        )
    )
    messages = [{"role": "user", "content": "Test message"}]
    responses = await mock_service.chat_completion(messages)
    assert responses[0].content == "Test response"
    mock_service.content_safety_eval.content_safety_check.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completion_with_format(
    fn_mock_service: Callable[[bool], AzureOpenAIService],
):
    mock_service = fn_mock_service(with_api_key=True)  # type: ignore
    mock_service.client.chat.completions = MagicMock()
    mock_service.client.chat.completions.parse = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Formatted response"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
    )
    messages = [{"role": "user", "content": "Test message"}]
    response_format = MagicMock()  # Mock the response format as needed
    responses = await mock_service.chat_completion_with_format(
        messages, response_format
    )
    assert responses[0].content == "Formatted response"


@pytest.mark.asyncio
async def test_chat_completion_with_format_err(
    fn_mock_service: Callable[[bool], AzureOpenAIService],
):
    mock_service = fn_mock_service(with_api_key=True)  # type: ignore
    mock_service.client.chat.completions = MagicMock()
    mock_service.client.chat.completions.parse = AsyncMock(
        side_effect=Exception("API error")
    )
    messages = [{"role": "user", "content": "Test message"}]
    response_format = MagicMock()  # Mock the response format as needed

    with pytest.raises(Exception):
        await mock_service.chat_completion_with_format(messages, response_format)
