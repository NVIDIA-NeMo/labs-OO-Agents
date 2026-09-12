# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing skill preferences for NOOA interactive workspaces."""

from pathlib import Path

from nooa.skill import Skill


class PersistingSkills(Skill):
    """Remember or forget skills for future NOOA TUI and ACP sessions.

    Preferences belong to this workspace's .nooa/settings.yaml. They apply to
    fresh agents in either client; other live agents retain their current state.
    This saves discovery and activation preferences, not Python objects or
    package installations. Ordinary self.skills.load/activate remains local.
    """

    def __init__(self, workspace: Path):
        super().__init__()
        self._workspace = workspace.expanduser().resolve()

    async def remember(self, skill_id: str, directory: str | None = None) -> str:
        """Activate a skill here and remember it for future workspace sessions.

        Use an exact ID from self.skills.discovered(). For a skill from a local
        repository, supply its directory so fresh agents can discover it too;
        relative paths resolve against this workspace. An already active skill
        can still be remembered. Installation is separate. A save failure raises
        an error describing any changes already applied to the live session.
        """
        if directory is not None:
            await self._run("add", directory)
        await self._run("activate", skill_id)
        return f"Remembered `{skill_id}` in {self._workspace / '.nooa' / 'settings.yaml'}."

    async def forget(self, skill_id: str) -> str:
        """Deactivate a skill here and disable its automatic workspace activation.

        Saves the same preference as /skills deactivate. Other live sessions
        keep their state; source directories, installed packages, and saved
        session data are retained. A save failure raises an error.
        """
        await self._run("deactivate", skill_id)
        return f"Forgot `{skill_id}` in {self._workspace / '.nooa' / 'settings.yaml'}."

    async def _run(self, action: str, value: str) -> None:
        from .controls import SkillsControl
        from .options import SessionOptions

        control = SkillsControl(
            self._agent,
            SessionOptions.load(self._workspace),
            workspace=self._workspace,
            command_registry=getattr(self._agent, "_command_registry", None),
        )
        result = await control.run([action, value])
        if not result.success or any(output.style == "warning" for output in result.outputs):
            raise RuntimeError(str(result))
