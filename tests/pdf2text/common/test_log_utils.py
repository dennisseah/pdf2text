import logging

import pytest

from pdf2text.common.log_utils import set_log_level


def test_set_log_level():
    logger = set_log_level("DEBUG")
    assert logger.level == logging.DEBUG
    logger.debug("This is a debug message.")

    logger = set_log_level("INFO")
    assert logger.level == logging.INFO

    logger = set_log_level("WARNING")
    assert logger.level == logging.WARNING

    logger = set_log_level("ERROR")
    assert logger.level == logging.ERROR
    logger.debug("This is a error message.")

    with pytest.raises(ValueError):
        set_log_level("INVALID_LEVEL")  # type: ignore
