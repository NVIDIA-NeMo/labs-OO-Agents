# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate YAML files."""

import sys

import yaml

retval = 0
for filename in sys.argv[1:]:
    try:
        with open(filename) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"{filename}: {e}")
        retval = 1
    except (UnicodeDecodeError, PermissionError):
        continue
sys.exit(retval)
