from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import AsyncIterator

from azure.ai.formrecognizer import AnalyzeResult
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from azure.identity.aio import DefaultAzureCredential
from lagom.environment import Env

from pdf2text.protocols.i_azure_form_recognizer import IAzureFormRecognizer


class AzureFormRecognizerEnv(Env):
    azure_form_recognizer_endpoint: str


@dataclass
class AzureFormRecognizer(IAzureFormRecognizer):
    env: AzureFormRecognizerEnv
    logger: Logger

    @asynccontextmanager
    async def get_client(self) -> AsyncIterator[DocumentAnalysisClient]:
        client: DocumentAnalysisClient | None = None
        credential = DefaultAzureCredential()

        try:
            client = DocumentAnalysisClient(
                self.env.azure_form_recognizer_endpoint,
                credential=credential,  # type: ignore
            )
            yield client
        finally:
            if client is not None:
                try:
                    await client.close()  # type: ignore
                except Exception as e:
                    self.logger.warning(f"Error closing client: {e}")

            try:
                await credential.close()
            except Exception as e:
                self.logger.warning(f"Error closing credential: {e}")

    async def analyze_document(self, path: str | Path) -> AnalyzeResult:
        async with self.get_client() as client:
            with open(path, "rb") as f:
                # The service call is asynchronous
                poller = await client.begin_analyze_document(
                    model_id="prebuilt-document", document=f
                )

                return await poller.result()
