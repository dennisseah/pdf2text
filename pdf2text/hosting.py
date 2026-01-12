"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging
import os

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from pdf2text.common.log_utils import set_log_level
from pdf2text.protocols.i_azure_form_recognizer import IAzureFormRecognizer
from pdf2text.protocols.i_azure_openai_service import IAzureOpenAIService
from pdf2text.protocols.i_openai_content_evaluator import IOpenAIContentEvaluator

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def logger() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "ERROR")
    if log_level not in ["ERROR", "WARNING", "INFO", "DEBUG"]:
        log_level = "ERROR"
    return set_log_level(log_level)  # type: ignore


@dependency_definition(container, singleton=True)
def azure_form_recognizer() -> IAzureFormRecognizer:
    from pdf2text.services.azure_form_recognizer import (
        AzureFormRecognizer,
    )

    return container[AzureFormRecognizer]


@dependency_definition(container, singleton=True)
def openai_content_evaluator() -> IOpenAIContentEvaluator:
    from pdf2text.services.openai_content_evaluator import (
        OpenAIContentEvaluator,
    )

    return container[OpenAIContentEvaluator]


@dependency_definition(container, singleton=True)
def azure_openai_service() -> IAzureOpenAIService:
    from pdf2text.services.azure_openai_service import AzureOpenAIService

    return container[AzureOpenAIService]
