import logging


def get_logger(name: str):
    """Get or create a logger with the given name."""
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate logs
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
