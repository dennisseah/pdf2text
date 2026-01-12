import asyncio
from pathlib import Path

from agent import get_drivers
from pdf2text.common.form_recognizer_parse import parse
from pdf2text.hosting import container
from pdf2text.protocols.i_azure_form_recognizer import IAzureFormRecognizer

current_path = Path(__file__).parent
sample = current_path / "test_data" / "ast_sci_data_tables_sample.pdf"


async def main() -> None:
    svc = container[IAzureFormRecognizer]
    result = await svc.analyze_document(sample)
    text = "\n".join(parse(result, tbl_format="grid"))
    if result:
        print(text)

    await get_drivers(text)


if __name__ == "__main__":
    asyncio.run(main())
