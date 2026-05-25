"""Tee-style logging helper (from ad_opt)."""

import atexit
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


def setup_tee_logging(
    *,
    log_file: Optional[str],
    default_log_dir: str = "logs",
    default_log_prefix: str = "run",
) -> Optional[Path]:
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(default_log_dir) / f"{default_log_prefix}_{ts}.log"
    else:
        log_file = str(log_file).strip()
        if not log_file:
            return None
        log_path = Path(log_file)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8", newline="\n")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = TeeStream(orig_stdout, fh)
    sys.stderr = TeeStream(orig_stderr, fh)

    def _cleanup():
        try:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        except Exception:
            pass
        try:
            fh.flush()
            fh.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    return log_path
