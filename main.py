import asyncio
from pathlib import Path

from agent import get_drivers, who_is_fastest
from pdf2text.common.form_recognizer_parse import parse
from pdf2text.hosting import container
from pdf2text.protocols.i_azure_form_recognizer import IAzureFormRecognizer

current_path = Path(__file__).parent
sample = current_path / "test_data" / "ast_sci_data_tables_sample.pdf"

# using Azure Form Recognizer to extract text from PDF
# call LLM functions to get drivers and identify fastest driver


async def main() -> None:
    svc = container[IAzureFormRecognizer]
    result = await svc.analyze_document(sample)
    text = "\n".join(parse(result, tbl_format="grid"))
    if result:
        print(text)

    await get_drivers(text)
    await who_is_fastest(text)


if __name__ == "__main__":
    asyncio.run(main())
