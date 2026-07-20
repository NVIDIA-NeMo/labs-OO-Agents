# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental strategies — not actively maintained.

These strategies are importable but not part of the recommended API.
Importing any of them is silent; instantiating emits a FutureWarning.

Usage:
    from nooa.experimental import PurePythonStrategy
    strategy = PurePythonStrategy()  # FutureWarning emitted here
"""

import warnings
from typing import Any


def _warn_experimental(name: str) -> None:
    warnings.warn(
        f"{name} is experimental and not actively maintained. "
        f"Use CodeActStrategy or PredictStrategy instead. "
        f"Experimental strategies may be removed in a future release.",
        FutureWarning,
        stacklevel=3,
    )


def PurePythonStrategy(*args: Any, **kwargs: Any) -> Any:
    """Create a PurePythonStrategy instance (experimental, emits FutureWarning)."""
    _warn_experimental("PurePythonStrategy")
    from nooa.strategies.pure_python import PurePythonStrategy as _Cls

    return _Cls(*args, **kwargs)


def CodeActLiteStrategy(*args: Any, **kwargs: Any) -> Any:
    """Create a CodeActLiteStrategy instance (experimental, emits FutureWarning)."""
    _warn_experimental("CodeActLiteStrategy")
    from nooa.strategies.codeact_lite import CodeActLiteStrategy as _Cls

    return _Cls(*args, **kwargs)


def ReflexionStrategy(*args: Any, **kwargs: Any) -> Any:
    """Create a ReflexionStrategy instance (experimental, emits FutureWarning)."""
    _warn_experimental("ReflexionStrategy")
    from nooa.strategies.reflexion import ReflexionStrategy as _Cls

    return _Cls(*args, **kwargs)


__all__ = [
    "CodeActLiteStrategy",
    "PurePythonStrategy",
    "ReflexionStrategy",
]
