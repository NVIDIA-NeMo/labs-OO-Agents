# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF v1.7 trajectory export — event-driven, OTel-decoupled.

Public surface:

- :class:`Trajectory` and the rest of the v1.7 Pydantic schema in
  :mod:`nooa.atif.schema`. Importable by downstream tooling
  (CyberGym uploaders, dashboards) to validate and serialize
  trajectories.
- :func:`install_atif` and :func:`atif_scope` — activation API for
  attaching the exporter to an :class:`EventManager`.
"""

from nooa.atif.exporter import AtifExporter
from nooa.atif.install import atif_scope, enable_atif, install_atif
from nooa.atif.schema import (
    SCHEMA_VERSION,
    AgentSchema,
    ContentPart,
    FinalMetricsSchema,
    ImageSource,
    MetricsSchema,
    ObservationResultSchema,
    ObservationSchema,
    StepObject,
    SubagentTrajectoryRef,
    ToolCallSchema,
    Trajectory,
)

__all__ = [
    "SCHEMA_VERSION",
    "AgentSchema",
    "AtifExporter",
    "ContentPart",
    "FinalMetricsSchema",
    "ImageSource",
    "MetricsSchema",
    "ObservationResultSchema",
    "ObservationSchema",
    "StepObject",
    "SubagentTrajectoryRef",
    "ToolCallSchema",
    "Trajectory",
    "atif_scope",
    "enable_atif",
    "install_atif",
]
