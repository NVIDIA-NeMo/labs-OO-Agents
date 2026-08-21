# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Contributed by Council of AI (CSOAI Ltd, UK 16939677) — https://councilof.ai
"""Make the example agent importable when pytest runs this directory directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
