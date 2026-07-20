# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type stub for _visibility.py.

Returning Literal[False] from __exit__ tells pyright that the context manager
never suppresses exceptions, so the with-block body is always fully executed
before control reaches code after the block. This prevents pyright from
flagging names defined inside ``with hidden:`` as "possibly unbound".
"""

import types
from typing import Any, Literal

class _Hidden:
    def __call__(self, func: Any) -> Any: ...
    def __enter__(self) -> _Hidden: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]: ...

hidden: _Hidden

def is_hidden_method(func: Any) -> bool: ...
def is_hidden_field(cls: Any, name: str) -> bool: ...
def is_hidden_module_variable(module: types.ModuleType, name: str) -> bool: ...
def filter_module_globals(module: types.ModuleType) -> dict[str, Any]: ...
def iter_agent_mro_modules(agent_class: type) -> list[types.ModuleType]: ...
def filter_mro_module_globals(agent_class: type) -> dict[str, Any]: ...
