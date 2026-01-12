import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from azure.ai.formrecognizer import (
    AnalyzeResult,
)
from tabulate import tabulate

from pdf2text.common.form_recognizer_parse import (
    TableData,
    format_table,
    in_table,
    parse,
)

current_path = Path(__file__).parent
data_doc = current_path / "data" / "form_recognizer_doc.json"


def test_table_data_to_csv() -> None:
    tbl_data = TableData(
        rows=[["A1", "B1"], ["A2", "B2"], []],
        headers=["HeaderA", "HeaderB"],
        span_offsets={1: [0, 10], 2: [20, 30]},
    )

    csv_output = tbl_data.to_csv()
    expected_csv = "\nHeaderA,HeaderB\nA1,B1\nA2,B2\n\n"
    assert csv_output == expected_csv


def test_table_data_to_json() -> None:
    tbl_data = TableData(
        rows=[["A1", "B1"], ["A2", "B2"], []],
        headers=["HeaderA", "HeaderB"],
        span_offsets={1: [0, 10], 2: [20, 30]},
    )

    expected_output = [
        {"HeaderA": "A1", "HeaderB": "B1"},
        {"HeaderA": "A2", "HeaderB": "B2"},
    ]

    csv_output = tbl_data.to_json()
    expected_csv = f"\n{json.dumps(expected_output)}\n"
    assert csv_output == expected_csv


def test_table_data_to_grid() -> None:
    tbl_data = TableData(
        rows=[["A1", "B1"], ["A2", "B2"], []],
        headers=["HeaderA", "HeaderB"],
        span_offsets={1: [0, 10], 2: [20, 30]},
    )

    grid_output = tbl_data.to_grid()
    expected_output = [
        {"HeaderA": "A1", "HeaderB": "B1"},
        {"HeaderA": "A2", "HeaderB": "B2"},
    ]
    assert (
        grid_output
        == f"\n{tabulate(expected_output, headers='keys', tablefmt='grid')}\n"
    )


@patch.object(TableData, "to_json")
@patch.object(TableData, "to_csv")
@patch.object(TableData, "to_grid")
def test_format_output(
    mock_to_grid: MagicMock, mock_to_csv: MagicMock, mock_to_json: MagicMock
) -> None:
    tbl_data = TableData(
        rows=[["A1", "B1"], ["A2", "B2"], []],
        headers=["HeaderA", "HeaderB"],
        span_offsets={1: [0, 10], 2: [20, 30]},
    )

    def reset_mocks() -> None:
        mock_to_csv.reset_mock()
        mock_to_json.reset_mock()
        mock_to_grid.reset_mock()

    tbl_data.format_output("csv")
    mock_to_csv.assert_called_once()
    mock_to_json.assert_not_called()
    mock_to_grid.assert_not_called()

    reset_mocks()

    tbl_data.format_output("json")
    mock_to_json.assert_called_once()
    mock_to_csv.assert_not_called()
    mock_to_grid.assert_not_called()

    reset_mocks()

    tbl_data.format_output("grid")
    mock_to_grid.assert_called_once()
    mock_to_csv.assert_not_called()
    mock_to_json.assert_not_called()


def test_format_table() -> None:
    with open(data_doc, "r") as f:
        tbl_json = json.load(f)
        result = AnalyzeResult.from_dict(tbl_json)
        tbl = result.tables[0]  # type: ignore

        data = format_table(tbl)
        assert data.headers == ["Column 1", "Column 2"]
        assert data.rows == [["row1 col1", "row1 col2"], ["row2 col1", "row2 col2"]]
        assert data.span_offsets == {1: [1000, 2000, 3000, 4000, 5000, 6000]}


def test_format_table_err() -> None:
    with open(data_doc, "r") as f:
        tbl_json = json.load(f)
        result = AnalyzeResult.from_dict(tbl_json)
        tbl = result.tables[0]  # type: ignore
        tbl.bounding_regions = []

        with pytest.raises(ValueError, match="Table has no bounding regions"):
            format_table(tbl)


def test_in_table() -> None:
    tbl1 = TableData(
        headers=["H1", "H2"],
        rows=[["R1C1", "R1C2"]],
        span_offsets={1: [0, 10], 2: [20]},
    )
    tbl2 = TableData(
        headers=["H3", "H4"],
        rows=[["R2C1", "R2C2"]],
        span_offsets={1: [30], 3: [40]},
    )

    tables = [tbl1, tbl2]

    result_tbl = in_table(1, 10, tables)
    assert result_tbl == tbl1

    result_tbl = in_table(2, 20, tables)
    assert result_tbl == tbl1

    result_tbl = in_table(1, 30, tables)
    assert result_tbl == tbl2

    result_tbl = in_table(3, 40, tables)
    assert result_tbl == tbl2

    result_tbl = in_table(2, 999, tables)
    assert result_tbl is None


def test_parse() -> None:
    with open(data_doc, "r") as f:
        tbl_json = json.load(f)
        result = AnalyzeResult.from_dict(tbl_json)

        parsed_output = parse(result)
        assert len(parsed_output) > 0
        assert parsed_output == [
            "content 1",
            "content 2",
            "\nColumn 1,Column 2\nrow1 col1,row1 col2\nrow2 col1,row2 col2\n\n",
            "after table content",
        ]
