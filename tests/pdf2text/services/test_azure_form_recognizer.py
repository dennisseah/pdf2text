from contextlib import asynccontextmanager
from logging import Logger
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.formrecognizer import AnalyzeResult
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from pytest_mock import MockerFixture

from pdf2text.services.azure_form_recognizer import (
    AzureFormRecognizer,
    AzureFormRecognizerEnv,
)

current_path = Path(__file__).parent.parent
data_doc = current_path / "common" / "data" / "form_recognizer_doc.json"


@pytest.fixture
def mock_env() -> AzureFormRecognizerEnv:
    return AzureFormRecognizerEnv(
        azure_form_recognizer_endpoint="https://mock-docs.cognitiveservices.azure.com/"
    )


@pytest.mark.asyncio
async def test_get_client(mocker: MockerFixture, mock_env: AzureFormRecognizerEnv):
    service = AzureFormRecognizer(env=mock_env, logger=MagicMock())
    mocker.patch(
        "pdf2text.services.azure_form_recognizer.DocumentAnalysisClient",
        return_value=AsyncMock(),
    )
    async with service.get_client() as client:  # type: ignore
        assert client is not None


@pytest.fixture
@asynccontextmanager
async def mock_client(mocker: MockerFixture) -> AsyncIterator[MockerFixture]:
    mock_client = mocker.MagicMock(spec=DocumentAnalysisClient)
    mock_poller = AsyncMock()
    mock_poller.result = AsyncMock(return_value=MagicMock(spec=AnalyzeResult))

    with open(data_doc, "r") as f:
        mock_open = mocker.mock_open(read_data=f.read())
        mocker.patch("builtins.open", mock_open)

    async with mock_client:
        mock_client.begin_analyze_document = AsyncMock()
        yield mock_client


@pytest.mark.asyncio
async def test_client_close_err(mocker: MockerFixture):
    mock_client = mocker.MagicMock(spec=DocumentAnalysisClient)
    mock_client.close = AsyncMock(side_effect=Exception("Close error"))
    mock_logger = MagicMock(spec=Logger)
    mock_logger.warning = MagicMock()

    mocker.patch(
        "pdf2text.services.azure_form_recognizer.DocumentAnalysisClient",
        return_value=mock_client,
    )
    service = AzureFormRecognizer(env=MagicMock(), logger=mock_logger)
    async with service.get_client() as client:  # type: ignore
        assert client is not None
    mock_logger.warning.assert_called_once_with("Error closing client: Close error")


@pytest.mark.asyncio
async def test_cred_close_err(mocker: MockerFixture):
    mock_client = mocker.MagicMock(spec=DocumentAnalysisClient)
    mock_logger = MagicMock(spec=Logger)
    mock_logger.warning = MagicMock()

    mocker.patch(
        "pdf2text.services.azure_form_recognizer.DocumentAnalysisClient",
        return_value=mock_client,
    )
    mocker.patch(
        "pdf2text.services.azure_form_recognizer.DefaultAzureCredential",
        return_value=MagicMock(close=AsyncMock(side_effect=Exception("Close error"))),
    )
    service = AzureFormRecognizer(env=MagicMock(), logger=mock_logger)
    async with service.get_client() as client:  # type: ignore
        assert client is not None
    mock_logger.warning.assert_called_once_with("Error closing credential: Close error")


@pytest.mark.asyncio
async def test_analyze_document(
    mock_env: AzureFormRecognizerEnv,
    mocker: MockerFixture,
    mock_client: MagicMock,
) -> None:
    service = AzureFormRecognizer(env=mock_env, logger=MagicMock())
    mocker.patch.object(AzureFormRecognizer, "get_client", return_value=mock_client)
    results = await service.analyze_document("test_secret")

    assert results is not None
