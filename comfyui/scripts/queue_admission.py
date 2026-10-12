"""Read-only, fail-closed admission shared by the two prompt submission routes."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def wait_for_empty_queue(base: str, deadline: float) -> None:
    """Require a fresh empty queue within a caller-owned monotonic deadline.

    This is cooperative admission, not an atomic reservation against other clients.
    The caller must finish local preparation first and POST immediately on return.
    """
    url = base.rstrip('/') + '/queue'
    request = urllib.request.Request(url, method='GET')
    transient = 0

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Queue admission timed out; NO_POST; prompt_id=None')
        return seconds

    while True:
        delay = 2
        try:
            with urllib.request.urlopen(request, timeout=min(15, remaining())) as response:
                queue = json.load(response)
        except urllib.error.HTTPError as exc:
            if not 500 <= exc.code < 600:
                body = exc.read().decode('utf-8', errors='replace')
                raise RuntimeError(f'Queue admission failed: HTTP {exc.code}: {body}; NO_POST') from exc
            transient += 1
            delay = min(20, 2 * transient)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            remaining()
            transient += 1
            delay = min(20, 2 * transient)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(f'Invalid queue admission response from {url}; NO_POST') from exc
        else:
            if (not isinstance(queue, dict) or
                    not isinstance(queue.get('queue_running'), list) or
                    not isinstance(queue.get('queue_pending'), list)):
                raise RuntimeError(f'Invalid queue admission shape from {url}; NO_POST')
            remaining()
            if not queue['queue_running'] and not queue['queue_pending']:
                return
            transient = 0
        time.sleep(min(delay, remaining()))
