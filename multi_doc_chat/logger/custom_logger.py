import os
import logging
from datetime import datetime
from pathlib import Path
import structlog

class CustomLogger:
    def __init__(self, logs_dir: str | Path | None = None, *, log_file_name: str | None = None):
        '''
        Initialize the CustomLogger.

        By default logs are written to the repository root `logs/` directory. You can
        override this by passing `logs_dir` (str or Path) or set an explicit
        `log_file_name`.
        '''
        # Default to repository root logs folder (two parents up from this file)
        default_root = Path(__file__).resolve().parents[2]
        self.logs_dir = Path(logs_dir) if logs_dir is not None else (default_root / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if log_file_name is None:
            log_file_name = f"app_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"

        self.log_path = str(self.logs_dir / log_file_name)

        # Ensure configuration happens only once per instance
        self._configured = False
        
    def get_logger(self, name=__file__):
        '''Get a structured logger instance with the specified name.'''
        logger_name = os.path.basename(name).replace(".py", "")

        # Configure logging and structlog once per instance
        if not self._configured:
            file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(message)s'))

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(message)s'))

            # basicConfig will configure the root logger if not already configured
            logging.basicConfig(
                level=logging.INFO,
                handlers=[file_handler, console_handler]
            )

            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    structlog.processors.add_log_level,
                    structlog.processors.EventRenamer(to="event"),
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            self._configured = True

        return structlog.get_logger(logger_name)

    # --- Backwards-compatible class-level convenience methods ---
    # Some modules import the class as `from ...custom_logger import CustomLogger as log`
    # and then call `log.info(...)`. Provide class-level forwarding methods so
    # those call sites continue to work without changing imports.
    _default_logger = None
    _default_instance = None

    @classmethod
    def _ensure_default(cls):
        if cls._default_logger is None:
            inst = cls()
            cls._default_instance = inst
            # Use module name as logger name for the default
            cls._default_logger = inst.get_logger(__file__)
        return cls._default_logger

    @classmethod
    def info(cls, *args, **kwargs):
        return cls._ensure_default().info(*args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        return cls._ensure_default().error(*args, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        return cls._ensure_default().warning(*args, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        return cls._ensure_default().debug(*args, **kwargs)

    @classmethod
    def critical(cls, *args, **kwargs):
        return cls._ensure_default().critical(*args, **kwargs)

    @classmethod
    def exception(cls, *args, **kwargs):
        return cls._ensure_default().exception(*args, **kwargs)