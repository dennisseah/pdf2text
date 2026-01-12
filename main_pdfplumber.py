import asyncio
from pathlib import Path

import pdfplumber

from agent import get_drivers

current_path = Path(__file__).parent
sample = current_path / "test_data" / "ast_sci_data_tables_sample.pdf"


def extract_text_pdfplumber():
    text = ""
    with pdfplumber.open(sample) as pdf:
        for page in pdf.pages:
            text += page.extract_text(layout=True) or ""
    return text


async def main() -> None:
    text = extract_text_pdfplumber()
    print(text)

    await get_drivers(text)


if __name__ == "__main__":
    asyncio.run(main())
