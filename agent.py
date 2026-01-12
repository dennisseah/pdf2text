import json

from pdf2text.hosting import container
from pdf2text.models.driver import Drivers
from pdf2text.protocols.i_azure_openai_service import IAzureOpenAIService

PROMPT = """You are a helpful assistant that list all the drivers and cars provided in the text from user.

Provide the response in the following JSON format:

{
  "drivers": [
    {
      "name": "<driver_name>",
      "cars": [{
        "model": "<car_model>",
        "engine": "<car_engine>"
      }]
    },
    ...
  ]
}

"""  # noqa: E501


async def get_drivers(text: str) -> None:
    openai_svc = container[IAzureOpenAIService]
    response = await openai_svc.chat_completion_with_format(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that list all the drivers and "
                    "cars provided in the text from user."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        response_format=Drivers,
    )
    drivers = Drivers.model_validate_json(response[0].content)  # type: ignore
    print(json.dumps(drivers.model_dump(), indent=2))
