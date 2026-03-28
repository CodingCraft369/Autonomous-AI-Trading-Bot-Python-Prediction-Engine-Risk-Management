"""
logging_monitor/logger.py
Windows-safe logger — fixes three bugs:

1. WinError 32 (PermissionError on log rotation):
   RotatingFileHandler uses os.rename() which fails on Windows when another
   process holds the file open.  Fix: use delay=True so the file is only
   opened when needed, and set the Windows-compatible rotation strategy
   using a subclass that catches the PermissionError and retries silently.

2. Duplicate log lines:
   app.py calls logging.basicConfig() which attaches a StreamHandler to the
   ROOT logger.  Every other module's logger propagates up to root, so each
   message prints twice.  Fix: disable propagation on all named loggers and
   configure the root logger to NullHandler.

3. Encoding errors on Windows console (UnicodeEncodeError on emoji):
   Fix: force UTF-8 + replace mode on the stream handler.
"""
import logging
import sys
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR   = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE  = LOG_DIR / "bot.log"

# ── Format ─────────────────────────────────────────────────────────────────
_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Windows-safe rotating file handler ────────────────────────────────────
# Standard RotatingFileHandler fails on Windows with WinError 32 because
# os.rename() cannot rename a file that another process has open.
# This subclass catches that and silently continues writing to the same file
# until the next rotation attempt succeeds.

class _WinSafeRotatingHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that silently skips rotation on WinError 32."""

    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            # Another process holds the log file — skip this rotation.
            # The file will be rotated on the next attempt.
            pass
        except OSError as e:
            if getattr(e, "winerror", None) == 32:
                pass  # WinError 32 — file in use, skip rotation
            else:
                raise


# Import after defining the subclass (handlers module may not be imported yet)
import logging.handlers  # noqa: E402 — needed for the class above


# ── Shared file handler (one instance, shared across all loggers) ──────────
# Sharing one handler avoids the race condition where multiple loggers all
# try to rotate simultaneously and fight over the file lock.
_file_handler: logging.Handler | None = None


def _get_file_handler() -> logging.Handler:
    global _file_handler
    if _file_handler is None:
        _file_handler = _WinSafeRotatingHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=3,
            encoding="utf-8",
            delay=True,                   # only open file on first write
        )
        _file_handler.setFormatter(_FMT)
        _file_handler.setLevel(LOG_LEVEL)
    return _file_handler


def _get_console_handler() -> logging.Handler:
    """UTF-8 console handler — safe on Windows even without a UTF-8 terminal."""
    try:
        stream = open(
            sys.stdout.fileno(), mode="w", encoding="utf-8",
            errors="replace", closefd=False, buffering=1,
        )
    except Exception:
        stream = sys.stdout
    ch = logging.StreamHandler(stream)
    ch.setFormatter(_FMT)
    ch.setLevel(LOG_LEVEL)
    return ch


# ── Silence the root logger so basicConfig() in app.py doesn't cause duplicates
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger().setLevel(logging.WARNING)   # root only shows warnings+


# ── Public API ──────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.  Idempotent — safe to call multiple times
    with the same name.  Each logger:
      • Writes to console (UTF-8, WinError-safe)
      • Writes to logs/bot.log via a single shared rotating file handler
      • Does NOT propagate to root (prevents duplicate lines from basicConfig)
    """
    logger = logging.getLogger(name)

    # Already configured — return immediately
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False          # ← key: stop propagation to root

    logger.addHandler(_get_console_handler())
    logger.addHandler(_get_file_handler())

    return logger