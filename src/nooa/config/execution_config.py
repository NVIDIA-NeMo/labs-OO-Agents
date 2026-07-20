# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict


class ExecutionConfig(BaseModel):
    """Framework-level execution guards. Set at Agent class level:

    class MyAgent(Agent, execution=ExecutionConfig(max_nesting_depth=5)):
        ...
    """

    model_config = ConfigDict(frozen=True)

    max_nesting_depth: int = 10  # max agent-in-agent recursion depth

    def merge_with(self, other: "ExecutionConfig") -> "ExecutionConfig":
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: ExecutionConfig(field=value)."
            )
        return self.model_copy(update={k: getattr(other, k) for k in other.model_fields_set})
