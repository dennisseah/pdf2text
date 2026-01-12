import logging
from typing import Literal


def set_log_level(
    log_level: Literal["ERROR", "WARNING", "INFO", "DEBUG"],
) -> logging.Logger:
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger("pdf2text")
    logger.setLevel(log_level)

    # Create console handler with conditional coloring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Custom formatter with conditional colors
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            color_code = "\033[91m" if record.levelname == "ERROR" else "\033[94m"
            reset_code = "\033[0m"
            formatted = super().format(record)
            return f"{color_code}{formatted}{reset_code}"

    formatter = ColoredFormatter(fmt=" %(name)s :: %(levelname)-8s :: %(message)s")
    console_handler.setFormatter(formatter)

    # Add handler to logger if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
