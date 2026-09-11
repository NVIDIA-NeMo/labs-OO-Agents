# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare identical stopped-session copies for native/ACP acceptance testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from nooa.sessions import SessionStore


def prepare_sessions(source: Path, output: Path) -> dict:
    """Copy a session under its exclusive lock, including session-scoped memory.

    SQLite backup includes committed WAL contents. The second copy comes from
    the first backup, so both hosts start with identical bytes. Never overwrite
    an existing acceptance directory or copy a live session.
    """
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if source.suffix != ".db":
        raise ValueError("Source must be a NOOA session .db file")
    with SessionStore(source.parent).open(source.stem) as handle:
        output.mkdir(parents=True, exist_ok=False)
        try:
            native = output / "native"
            pool = output / "pool"
            native.mkdir()
            pool.mkdir()
            databases = [source]
            memory = source.with_name(f"{source.stem}-memory.db")
            if memory.is_file():
                databases.append(memory)
            hashes = {}
            for database in databases:
                target = native / database.name
                connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
                backup = sqlite3.connect(target)
                try:
                    connection.backup(backup)
                finally:
                    backup.close()
                    connection.close()
                shutil.copyfile(target, pool / database.name)
                hashes[database.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest = {
                "session_id": handle.id,
                "workspace": handle.info.working_directory,
                "model": handle.info.model,
                "source": str(source),
                "native_sessions": str(native),
                "pool_sessions": str(pool),
                "sha256": hashes,
                "shared_resources": "Workspace files, project memory and user configuration are shared.",
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            return manifest
        except BaseException:
            shutil.rmtree(output)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Stopped session .db file")
    parser.add_argument("--output", required=True, type=Path, help="New acceptance directory")
    args = parser.parse_args()
    print(json.dumps(prepare_sessions(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
