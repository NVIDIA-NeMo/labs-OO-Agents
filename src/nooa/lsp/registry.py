# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for LSP server configurations."""

from pydantic import BaseModel


class LSPServerConfig(BaseModel):
    """Configuration for an LSP server."""
    command: list[str]
    extensions: list[str]


class LSPServerRegistry:
    """Registry to look up LSP servers based on file extensions."""

    def __init__(self):
        self._servers: list[LSPServerConfig] = []
        
        # Register standard language servers
        self.register(
            LSPServerConfig(
                command=["pyright-langserver", "--stdio"],
                extensions=[".py"],
            )
        )
        self.register(
            LSPServerConfig(
                command=["typescript-language-server", "--stdio"],
                extensions=[".ts", ".js", ".tsx", ".jsx"],
            )
        )
        self.register(
            LSPServerConfig(
                command=["rust-analyzer"],
                extensions=[".rs"],
            )
        )
        self.register(
            LSPServerConfig(
                command=["gopls"],
                extensions=[".go"],
            )
        )
        self.register(
            LSPServerConfig(
                command=["clangd"],
                extensions=[".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"],
            )
        )

    def register(self, config: LSPServerConfig):
        """Register a new language server."""
        self._servers.append(config)

    def get_server_for_extension(self, extension: str) -> LSPServerConfig | None:
        """Look up the appropriate language server for a given extension (e.g. '.py')."""
        for config in self._servers:
            if extension in config.extensions:
                return config
        return None
