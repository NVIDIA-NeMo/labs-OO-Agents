# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc.adapters — doc() adapters for third-party libraries.

Each module adapts a popular library's documentation to agentdoc's
ModuleInfo/TypeInfo format, replacing auto-generated noise with a
curated, token-efficient view.

Import the adapter you need before calling doc():

    import nooa.agentdoc.adapters.plotly  # registers extractors for plotly

Or register all adapters for installed libraries at once:

    from nooa.agentdoc.adapters import register_all
    register_all()

Available adapters:
    plotly — plotly, plotly.express, plotly.graph_objects
    pandas — pandas.DataFrame, pandas.Series
"""


def register_all():
    """Register adapters for all installed libraries.

    Silently skips any library that is not installed.

    Returns:
        List of successfully registered library names.
    """
    registered = []

    try:
        import nooa.agentdoc.adapters.plotly as _  # noqa: F401

        registered.append("plotly")
    except ImportError:
        pass

    try:
        import nooa.agentdoc.adapters.pandas as _  # noqa: F401

        registered.append("pandas")
    except ImportError:
        pass

    return registered


__all__ = ["register_all"]
