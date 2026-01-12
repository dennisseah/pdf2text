from pathlib import Path
from typing import Protocol

from azure.ai.formrecognizer import AnalyzeResult


class IAzureFormRecognizer(Protocol):
    async def analyze_document(self, path: str | Path) -> AnalyzeResult:
        """
        Analyze a document using Azure Form Recognizer.

        :param path: The file path to the document to be analyzed.
        :return: An AnalyzeResult object containing the analysis results.
        """
        ...
