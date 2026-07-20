# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enable running as: python -m eval_pipeline"""

# Enable faulthandler early for debugging hangs
# - Dumps Python traceback on SIGSEGV, SIGFPE, SIGABRT, SIGBUS
# - Send SIGUSR1 to dump traceback on demand: kill -USR1 <pid>
import faulthandler
import signal
import sys

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

from .cli import main  # noqa: E402 - faulthandler must be enabled first

if __name__ == "__main__":
    main()
