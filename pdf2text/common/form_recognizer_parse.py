import json
from pathlib import Path
from typing import Literal

import pandas as pd
from azure.ai.formrecognizer import AnalyzeResult, DocumentTable
from pydantic import BaseModel, Field
from tabulate import tabulate

current_folder = Path(__file__).parent
document_path = current_folder.parent.parent / "x.json"


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]]
    span_offsets: dict[int, list[int]]
    added: bool = False

    def format_output(self, tbl_format: Literal["csv", "json", "grid"]) -> str:
        if tbl_format == "csv":
            return self.to_csv()
        elif tbl_format == "grid":
            return self.to_grid()
        else:
            return self.to_json()

    def to_csv(self) -> str:
        if not self.rows[-1]:
            self.rows.pop()

        df = pd.DataFrame(self.rows, columns=self.headers)  # type: ignore
        return f"\n{df.to_csv(index=False)}\n"

    def to_json(self) -> str:
        if not self.rows[-1]:
            self.rows.pop()

        df = pd.DataFrame(self.rows, columns=self.headers)  # type: ignore
        data = df.to_dict(orient="records")
        return f"\n{json.dumps(data)}\n"

    def to_grid(self) -> str:
        if not self.rows[-1]:
            self.rows.pop()

        df = pd.DataFrame(self.rows, columns=self.headers)  # type: ignore
        data = df.to_dict(orient="records")
        result = tabulate(data, headers="keys", tablefmt="grid")
        return f"\n{result}\n"


def format_table(tbl: DocumentTable) -> TableData:
    if not tbl.bounding_regions:
        raise ValueError("Table has no bounding regions")

    span_offsets: dict[int, list[int]] = {}
    for cell in tbl.cells:
        if cell.spans and cell.bounding_regions:
            pg_num = cell.bounding_regions[0].page_number
            if pg_num not in span_offsets:
                span_offsets[pg_num] = []
            span_offsets[pg_num].append(cell.spans[0].offset)

    tbl_data = TableData(
        rows=[[] for _ in range(tbl.row_count)],
        span_offsets=span_offsets,
    )

    for cell in tbl.cells:
        if cell.kind == "columnHeader":
            tbl_data.headers.append(cell.content)
        elif cell.kind == "content":
            row_idx = cell.row_index - 1
            tbl_data.rows[row_idx].append(cell.content)

    return tbl_data


def in_table(
    page_num: int, span_offset: int, tables: list[TableData]
) -> TableData | None:
    for tbl in tables:
        if page_num in tbl.span_offsets:
            offsets = tbl.span_offsets[page_num]
            if span_offset in offsets:
                return tbl

    return None


def parse(
    result: AnalyzeResult, tbl_format: Literal["csv", "json", "grid"] = "csv"
) -> list[str]:
    results: list[str] = []
    table_data = []

    paragraphs = result.paragraphs
    tables = result.tables
    if tables:
        for tbl in tables:
            table_data.append(format_table(tbl))

    if paragraphs:
        for p in paragraphs:
            if (
                p.bounding_regions
                and p.spans
                and p.role not in ["pageNumber", "pageFooter"]
            ):
                tbl = in_table(
                    p.bounding_regions[0].page_number, p.spans[0].offset, table_data
                )
                if tbl:
                    if not tbl.added:
                        results.append(tbl.format_output(tbl_format))
                        tbl.added = True
                else:
                    results.append(p.content)

    return results
