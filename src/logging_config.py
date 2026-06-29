"""
Structured logging configuration for Vault Copilot.
Replaces all print() calls with loguru structured logging.
"""
import sys
from loguru import logger

from src.config import LOG_LEVEL, LOG_JSON


def setup_logging():
    """Configure loguru with structured output."""
    logger.remove()  # Remove default handler

    if LOG_JSON:
        # Machine-readable JSON logs (production)
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            serialize=True,
        )
    else:
        # Human-readable colored logs (development)
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    # File logging with rotation (always JSON)
    logger.add(
        "logs/vault_copilot.log",
        level="DEBUG",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    return logger


# Initialize on import
setup_logging()
