"""Structured logging setup.

Emits one JSON object per line to stdout — easy for any log aggregator
(CloudWatch, Render/Railway's log viewer, Datadog, etc.) to parse without
custom parsing rules. Never log request bodies, situation text, or API
keys: only metadata (method, path, status, duration, error type).
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Any structured fields passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info") or key in payload:
                continue
            if key.startswith("_") or not isinstance(value, (str, int, float, bool, type(None))):
                continue
            if key in (
                "name",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "taskName",
            ):
                continue
            payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Keep noisy third-party access logs at a sane level.
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("google_genai").setLevel("WARNING")
