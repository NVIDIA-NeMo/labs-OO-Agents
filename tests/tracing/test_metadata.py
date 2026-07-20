# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for git and environment metadata capture."""

import subprocess
from unittest.mock import patch

from nooa.tracing._metadata import (
    get_all_metadata,
    get_environment_metadata,
    get_git_metadata,
)


class TestGitMetadata:
    """Test git metadata capture."""

    def test_get_git_metadata_success(self):
        """Test successful git metadata capture."""
        with (
            patch("subprocess.check_output") as mock_check_output,
            patch("subprocess.call") as mock_call,
        ):
            # Mock git commands
            mock_check_output.side_effect = [
                b"a1b2c3d4e5f6789012345678901234567890abcd\n",  # git rev-parse HEAD
                b"main\n",  # git rev-parse --abbrev-ref HEAD
            ]
            mock_call.return_value = 1  # git diff --quiet (dirty state)

            metadata = get_git_metadata()

            assert metadata["git.commit"] == "a1b2c3d"
            assert metadata["git.commit_full"] == "a1b2c3d4e5f6789012345678901234567890abcd"
            assert metadata["git.dirty"] == "true"
            assert metadata["git.branch"] == "main"

    def test_get_git_metadata_clean_repo(self):
        """Test git metadata with clean repo."""
        with (
            patch("subprocess.check_output") as mock_check_output,
            patch("subprocess.call") as mock_call,
        ):
            mock_check_output.side_effect = [
                b"abc123\n",
                b"feature/test\n",
            ]
            mock_call.return_value = 0  # git diff --quiet (clean)

            metadata = get_git_metadata()

            assert metadata["git.dirty"] == "false"
            assert metadata["git.branch"] == "feature/test"

    def test_get_git_metadata_not_in_repo(self):
        """Test git metadata when not in a git repository."""
        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = subprocess.CalledProcessError(128, "git")

            metadata = get_git_metadata()

            assert metadata == {}

    def test_get_git_metadata_git_not_installed(self):
        """Test git metadata when git is not installed."""
        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = FileNotFoundError()

            metadata = get_git_metadata()

            assert metadata == {}


class TestEnvironmentMetadata:
    """Test environment metadata capture."""

    def test_get_environment_metadata(self):
        """Test environment metadata capture."""
        with (
            patch("platform.python_version") as mock_py_version,
            patch("platform.node") as mock_node,
        ):
            mock_py_version.return_value = "3.12.0"
            mock_node.return_value = "test-machine"

            metadata = get_environment_metadata()

            assert "nooa.version" in metadata
            assert metadata["python.version"] == "3.12.0"
            assert metadata["hostname"] == "test-machine"


class TestAllMetadata:
    """Test combined metadata capture."""

    def test_get_all_metadata(self):
        """Test that all metadata is combined correctly."""
        with (
            patch("nooa.tracing._metadata.get_git_metadata") as mock_git,
            patch("nooa.tracing._metadata.get_environment_metadata") as mock_env,
        ):
            mock_git.return_value = {
                "git.commit": "abc123",
                "git.branch": "main",
            }
            mock_env.return_value = {
                "nooa.version": "0.1.0",
                "python.version": "3.12.0",
            }

            metadata = get_all_metadata()

            assert metadata["git.commit"] == "abc123"
            assert metadata["git.branch"] == "main"
            assert metadata["nooa.version"] == "0.1.0"
            assert metadata["python.version"] == "3.12.0"
