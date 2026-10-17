# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child-side probe: can a ``multiprocessing`` connection survive the container?

The broker protocol is built on ``multiprocessing.connection.Connection``. On
Windows that is a *named pipe* created by ``_winapi.CreateNamedPipe`` with a
default security descriptor, and the parent hands the child a duplicated
handle rather than letting it open the name.

If duplication is enough, the container never performs a name lookup and the
pipe DACL is irrelevant. If it is not, the whole broker transport needs an
anonymous-pipe replacement — a much larger change than a new process launcher.
That difference is why this is a measurement and not an assumption.

Reconstructs a ``PipeConnection`` from the numeric handle passed on argv,
echoes one frame, and exits. Any failure is reported in the reply rather than
raised, so the parent learns *why* instead of just seeing a dead child.
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    handle = int(sys.argv[1])
    payload = sys.argv[2]
    try:
        from multiprocessing.connection import PipeConnection

        conn = PipeConnection(handle, readable=True, writable=True)
        try:
            conn.send({"reply": payload.upper(), "ok": True})
            conn.close()
        except BaseException as exc:  # noqa: BLE001 - report, never crash
            # A send failure is itself the result; try one more channel only if
            # the pipe is still usable, which it usually is not.
            return 3 if isinstance(exc, OSError) else 4
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
