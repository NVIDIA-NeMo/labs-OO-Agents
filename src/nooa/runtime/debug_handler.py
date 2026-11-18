# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug signal handler for nooa (public path).

The implementation lives in :mod:`nooa._debug_handler`, outside
``nooa.runtime``, so ``import nooa`` can install the handler without loading
the runtime package. This module is an alias of it: both names refer to the
same module object, so module state and ``unittest.mock.patch`` targets are
shared.
"""

import sys
from typing import TYPE_CHECKING

from nooa import _debug_handler

if TYPE_CHECKING:  # the same names, for type checkers and IDEs
    from nooa._debug_handler import *  # noqa: F403

sys.modules[__name__] = _debug_handler
